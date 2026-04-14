"""
MAD (Multi-Agent Debate) 에세이 자동 평가 파이프라인
Critic → Defender(critic 결과 참조) → Final Judge 3단계 구조
[변경] Sequential MAD: Defender가 Critic 결과를 보고 반박하는 구조로 수정
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from openai import OpenAI, RateLimitError

# =========================
# 로깅 설정
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# =========================
# 설정
# =========================
@dataclass
class Config:
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model_name: str = "gpt-4o-mini"
    essay_data_dir: str = "data/selected_prompt_jsons"
    input_dirs: list[str] = field(default_factory=lambda: ["inference_results/gemma", "inference_results/qwen", "inference_results/llama", "inference_results/gpt"])
    output_dirs: list[str] = field(default_factory=lambda: ["judge_results/exp03_mad2/gemma", "judge_results/exp03_mad2/qwen", "judge_results/exp03_mad2/llama", "judge_results/exp03_mad2/gpt"])   
    max_retries: int = 3
    retry_base_wait: float = 5.0
    domain_sleep: float = 1.0
    agent_sleep: float = 0.7

    def validate(self):
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        if len(self.input_dirs) != len(self.output_dirs):
            raise ValueError("input_dirs와 output_dirs의 길이가 일치하지 않습니다.")


# =========================
# 시스템 프롬프트
# =========================
SCORE_RUBRIC = """
[점수 기준 — 반드시 준수]

5점: 4개 기준 모두 결함 없음 + essay_text에서 직접 인용 가능한 구체적 강점 2개 이상
4점: 경미한 약점 1개 이하, rationale이 essay_text와 직접 연결되며 설득력 있음
3점: 아래 중 하나에 해당할 때만 사용
     - Critic의 major_issue 중 일부는 rebutted, 일부는 accepted인 경우
     - rationale이 애매하게 맞고 애매하게 틀린 경우 (단, 이 경우 반드시 근거 명시)
2점: major_issue 1개 이상이 accepted이고 rationale이 predicted_score를 정당화하지 못함
1점: 아래 중 하나 이상 — 영역 혼동 / hallucination / 점수-설명 모순 / 근거 전무

[주의]
- 4점은 "무난함"이 아니다. essay_text와의 직접 연결이 확인될 때만 부여하라.
- 5점은 구체적 인용 근거가 2개 이상 있을 때만 부여하라.
- 근거 없는 3점 금지. 근거 없는 4~5점도 동일하게 금지한다.
""".strip()

COMMON_CRITERIA = f"""
[평가 기준 정의]

반드시 아래 4개 기준을 기준으로 판단하라.

---

1. domain_match
- rationale이 target_domain의 평가 기준을 사용하고 있는가

판단 기준:
1. content
- 글의 주장과 핵심 내용이 문제에 적절하게 대응하는가
- 근거가 충분하고 구체적인가
- 주장과 근거 사이의 논리적 연결이 타당한가

2. organization
- 서론, 본론, 결론의 구조가 드러나는가
- 문단 간 연결이 자연스러운가
- 논리 전개 순서가 일관적인가

3. expression
- 문장이 자연스럽고 이해하기 쉬운가
- 어휘 사용이 적절한가
- 맞춤법, 띄어쓰기, 문법, 주술 호응에 문제가 없는가

감점 기준:
- 다른 영역 기준을 사용하면 감점
- 예: expression 평가인데 내용 논리만 설명 → domain mismatch

---

2. score_rationale_consistency
- predicted_score와 rationale이 서로 일치하는가

판단 기준:
- 높은 점수 → 강한 근거 필요
- 낮은 점수 → 명확한 결함 필요

감점 기준:
- 높은 점수인데 근거가 약함
- 낮은 점수인데 결함 설명 부족
- 점수와 설명이 모순됨

---

3. specificity
- rationale이 구체적인가

판단 기준:
- 실제 문장, 표현, 논지, 구조를 지적하는가

감점 기준:
- "전반적으로 좋다", "논리적이다" 같은 generic 표현만 사용
- 구체적 예시 없음

---

4. groundedness
- rationale이 essay_text에 근거하는가

판단 기준:
- 실제 텍스트에서 확인 가능한가

감점 기준:
- essay에 없는 내용을 만들어냄 (hallucination)
- 근거가 추상적이고 확인 불가

---

{SCORE_RUBRIC}
""".strip()




CRITIC_SYSTEM = f"""
[역할]
너는 rationale의 품질을 검증하는 심사위원이다.
평가 대상은 오직 rationale이다 — essay_text의 품질을 평가하지 마라.

목표:
- rationale에서 실제로 확인된 문제를 정확하게 지적하라
- 단, 아래 항목은 반드시 확인하고 해당하면 major_issue로 기록하라

{COMMON_CRITERIA}

---

[판단 규칙]

- 실제로 확인된 문제만 지적하라. 억지로 채우지 마라.
- 애매한 경우 minor_issue로만 처리하라.
- major_issue가 없으면 비워도 된다.

---

[major_issue 조건]
아래 중 실제로 해당하는 경우에만 기록하라.
단, ③④는 반드시 확인하고 해당하면 반드시 기록하라:

① groundedness 위반: essay_text에 없는 내용을 근거로 사용 (hallucination)
② domain mismatch: target_domain과 다른 기준으로 평가
③ 핵심 근거 부족: predicted_score가 3 이상인데 rationale에 구체적 문장/표현 인용이 전혀 없음
④ generic 전용: rationale 전체가 "논리적이다", "잘 썼다" 수준의 상투적 표현으로만 구성됨
   → 이 경우 specificity와 groundedness 모두 major_issue로 기록하라

---

[provisional_score 기준]
- major_issue 0개: 3~5점 (강점 확인 여부로 결정)
- major_issue 1개: 2~3점
- major_issue 2개 이상: 1~2점

---

[출력 형식]
{{
  "major_issues": [
    {{
      "criterion": "specificity",
      "reason": "predicted_score 4인데 구체적 문장 인용 전무"
    }}
  ],
  "minor_issues": [...],
  "provisional_score": 1~5
}}
""".strip()


DEFENDER_SYSTEM = f"""
[역할]
너는 Critic의 평가를 검토하고 rationale을 공정하게 방어하는 평가자이다.

[중요] 평가 대상은 rationale이다.
- essay_text가 좋더라도 rationale이 그것을 근거로 삼지 않았다면 방어 불가하다.
- 반박은 반드시 "rationale이 실제로 essay_text의 어떤 부분을 근거로 삼고 있는가"로만 가능하다.
- essay_text에서 새로운 근거를 발굴하여 rationale을 대신 방어하는 것을 금지한다.

목표:
- Critic의 각 major_issue를 rationale 텍스트에 근거하여 반박하거나 인정하라
- 반박할 수 없는 문제는 인정하고 remaining_concerns에 기록하라

{COMMON_CRITERIA}

---

[판단 규칙]

- Critic의 major_issues를 하나씩 검토하라
- 반박 조건: rationale 내에 해당 비판을 반증하는 구체적 표현이 실제로 존재할 때만 rebutted
- 아래 경우는 반드시 accepted 처리하라:
  - rationale에 구체적 인용/표현이 없는데 "있다"고 주장하는 경우
  - generic 비판인데 rationale도 generic한 경우

---

[provisional_score 기준]
- major_issue 전부 rebutted: 4~5점 검토 (단, rationale 자체의 구체성 확인 필수)
- major_issue 일부 accepted: 2~3점
- major_issue 전부 accepted: 1~2점

---

[출력 형식]
{{
  "critic_rebuttals": [
    {{
      "criterion": "specificity",
      "critic_claim": "구체적 인용 없음",
      "rebuttal": "rationale 내 '~라는 표현을 사용하여' 구절이 존재함 → 반박 성공",
      "result": "rebutted"
    }}
  ],
  "strengths": [...],
  "remaining_concerns": [...],
  "provisional_score": 1~5
}}
""".strip()


FINAL_JUDGE_SYSTEM = f"""
[역할]
Critic과 Defender의 의견을 종합하여 최종 점수를 결정하는 공정한 중재자이다.
평가 대상은 오직 rationale이다 — essay_text 자체의 품질을 평가하지 마라.

{COMMON_CRITERIA}

---

[핵심 원칙]
- 최종 판단 기준: "rationale이 essay_text에 근거하여 predicted_score를 얼마나 설득력 있게 정당화하는가"
- Defender의 반박이 "rationale 내 실제 표현"에 근거할 때만 rebutted로 인정하라
- Defender가 essay_text에서 새 근거를 발굴한 경우 → rebutted 무효, 해당 항목 accepted 처리
- "문제의 개수"가 아니라 "더 설득력 있는 근거가 어느 쪽에 있는가"로 결정하라

---

[가감점 규칙]

감점:
- Critic의 major_issue가 accepted인 경우 → 해당 항목 -1점
- rationale 전체가 generic → specificity, groundedness 각각 2점 이하
- domain mismatch 확인 → domain_match 2점 이하
- hallucination 확인 → groundedness 1점

가점:
- Defender가 rationale 내 실제 표현으로 반박 성공(rebutted) → 해당 major_issue 무효
  단, 이것만으로 자동 가점하지 말고 SCORE_RUBRIC 기준을 별도로 충족해야 4점 이상 부여

---

[판정 절차]
각 항목에 대해 반드시 아래 순서로 판단하라:
1. Critic의 major_issue 확인
2. Defender의 rebuttal result 확인
3. Defender가 "rationale 내 표현" 근거인지 vs "essay_text 새 발굴" 근거인지 판별
4. 가감점 규칙 적용
5. SCORE_RUBRIC 최종 검토 — 4점 이상은 essay_text와의 직접 연결이 rationale에 있을 때만

---

[출력 규칙]
- JSON 객체 하나만 출력하라. 코드블록 마크다운 사용 금지.
- 모든 점수는 1~5 정수.

[출력 형식]
{{
  "domain_match": 1~5,
  "score_rationale_consistency": 1~5,
  "specificity": 1~5,
  "groundedness": 1~5,
  "winner_side": {{
    "domain_match": "critic|defender|tie",
    "score_rationale_consistency": "critic|defender|tie",
    "specificity": "critic|defender|tie",
    "groundedness": "critic|defender|tie"
  }}
}}
""".strip()

DOMAINS = ["content", "organization", "expression"]


# =========================
# API 클라이언트
# =========================
class EssayJudgeClient:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.api_key)

    def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> Optional[dict]:
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    max_tokens=512,
                )
                return json.loads(response.choices[0].message.content)

            except RateLimitError:
                wait = self.config.retry_base_wait * (attempt + 1)
                logger.warning(f"RateLimit: {wait:.0f}s 대기 (시도 {attempt + 1}/{self.config.max_retries})")
                time.sleep(wait)

            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 실패: {e}")
                return None

            except Exception as e:
                logger.error(f"API 호출 오류: {e}")
                return None

        logger.error("최대 재시도 초과")
        return None


# =========================
# 프롬프트 빌더
# =========================
def build_user_prompt(sample: dict) -> str:
    return (
        f"[target_domain]\n{sample['target_domain']}\n\n"
        f"[predicted_score]\n{sample['predicted_score']}\n\n"
        f"[rationale]\n{sample['rationale']}\n\n"
        f"[essay_text]\n{sample['essay_text']}"
    )


# =========================
# Overall 점수 계산
# =========================
def compute_overall(result: dict) -> float:
    keys = ["domain_match", "score_rationale_consistency", "specificity", "groundedness"]
    scores = [result.get(k, 3) for k in keys]
    avg = sum(scores) / len(scores)
    min_score = min(scores)

    if min_score == 1:
        avg = min(avg, 2.0)
    elif min_score == 2:
        avg = min(avg, 3.0)

    return round(avg, 1)


# =========================
# MAD 파이프라인
# =========================
class MADPipeline:
    def __init__(self, client: EssayJudgeClient, agent_sleep: float = 0.7):
        self.client = client
        self.agent_sleep = agent_sleep

    def critic(self, sample: dict) -> Optional[dict]:
        return self.client.call_json(
            CRITIC_SYSTEM,
            build_user_prompt(sample) + "\n\n반드시 json으로만 응답하라.",
            temperature=0.1,
        )

    # [변경 2] defender가 critic_result를 인자로 받아 user_prompt에 포함
    def defender(self, sample: dict, critic_result: dict) -> Optional[dict]:
        user_prompt = (
            build_user_prompt(sample)
            + f"\n\n[critic_result]\n{json.dumps(critic_result, ensure_ascii=False)}"
            + "\n\nCritic의 각 지적 항목을 essay_text에 근거하여 반박하거나 인정하라."
            + "\n\n반드시 json으로만 응답하라."
        )
        return self.client.call_json(DEFENDER_SYSTEM, user_prompt, temperature=0.1)

    def final_judge(self, sample: dict, critic: dict, defender: dict) -> Optional[dict]:
        user_prompt = (
            build_user_prompt(sample)
            + f"\n\n[critic]\n{json.dumps(critic, ensure_ascii=False)}"
            + f"\n\n[defender]\n{json.dumps(defender, ensure_ascii=False)}"
            + "\n\n반드시 json으로만 응답하라."
        )
        result = self.client.call_json(FINAL_JUDGE_SYSTEM, user_prompt, temperature=0.0)
        if result:
            result["overall_judge"] = compute_overall(result)
        return result

    # [변경 3] run()에서 defender 호출 시 critic_result 전달
    def run(self, sample: dict) -> Optional[dict]:
        critic_result = self.critic(sample)
        if not critic_result:
            logger.warning(f"Critic 실패: essay_id={sample.get('essay_id', '?')}, domain={sample.get('target_domain')}")
            return None

        time.sleep(self.agent_sleep)

        defender_result = self.defender(sample, critic_result)  # critic_result 전달
        if not defender_result:
            logger.warning(f"Defender 실패: essay_id={sample.get('essay_id', '?')}, domain={sample.get('target_domain')}")
            return None

        time.sleep(self.agent_sleep)

        final_result = self.final_judge(sample, critic_result, defender_result)
        if not final_result:
            logger.warning(f"Final Judge 실패: essay_id={sample.get('essay_id', '?')}, domain={sample.get('target_domain')}")
            return None

        return final_result


# =========================
# Essay Index 빌더
# =========================
def build_essay_index(essay_data_dir: Path) -> dict:
    index = {}
    json_files = list(essay_data_dir.glob("*.json"))

    if not json_files:
        logger.warning(f"essay JSON 파일을 찾을 수 없습니다: {essay_data_dir}")
        return index

    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                essay_id = item.get("essay_id")
                if essay_id:
                    index[essay_id] = {
                        "prompt_text": item.get("prompt_text", "").strip(),
                        "essay_text": item.get("essay_text", "").strip(),
                    }
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"인덱스 로드 실패: {json_file} → {e}")

    logger.info(f"Essay index 구축 완료: {len(index)}개")
    return index


# =========================
# 체크포인트 관리
# =========================
def load_checkpoint(output_file: Path) -> set[str]:
    if not output_file.exists():
        return set()
    try:
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)
        done = {item["essay_id"] for item in data if "essay_id" in item}
        logger.info(f"체크포인트 로드: {len(done)}개 essay 처리 완료 ({output_file.name})")
        return done
    except Exception as e:
        logger.warning(f"체크포인트 로드 실패: {e}")
        return set()


def append_result(output_file: Path, result: dict, all_results: list):
    all_results.append(result)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


# =========================
# 단일 Essay 처리
# =========================
def process_essay(
    item: dict,
    essay_index: dict,
    pipeline: MADPipeline,
    sleep_sec: float,
) -> Optional[dict]:
    essay_id = item.get("essay_id")
    essay_data = essay_index.get(essay_id)

    if not essay_data:
        logger.warning(f"Essay 데이터 없음: {essay_id}")
        return None

    result_item = {
        "essay_id": essay_id,
        "source_id": essay_data.get("source_id"),
        "prompt_id": essay_data.get("prompt_id"),
        "status": "ok",
        "gold": essay_data.get("gold"),
        "prediction": item.get("prediction", {}),
        "judge": {}
    }

    for domain in DOMAINS:
        pred = item.get("prediction", {}).get(domain)
        if not pred:
            continue

        sample = {
            "essay_id": essay_id,
            "essay_text": essay_data["essay_text"],
            "target_domain": domain,
            "predicted_score": pred["score"],
            "rationale": pred["rationale"],
        }

        logger.info(f"처리 중: essay_id={essay_id}, domain={domain}")
        final_result = pipeline.run(sample)

        if final_result:
            result_item["judge"][domain] = final_result
        else:
            logger.warning(f"최종 결과 없음: essay_id={essay_id}, domain={domain}")
            result_item["judge"][domain] = {
                "domain_match": 3,
                "score_rationale_consistency": 3,
                "specificity": 3,
                "groundedness": 3,
                "overall_judge": 3.0
            }

        time.sleep(sleep_sec)

    if not result_item["judge"]:
        result_item["status"] = "fail"

    return result_item


# =========================
# 폴더 단위 처리
# =========================
def process_folder(
    input_folder: str,
    output_folder: str,
    essay_index: dict,
    pipeline: MADPipeline,
    config: Config,
):
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(exist_ok=True, parents=True)

    json_files = list(input_dir.glob("*.json"))
    logger.info(f"처리 대상 파일 수: {len(json_files)} ({input_dir})")

    for file in json_files:
        output_file = output_dir / file.name
        done_ids = load_checkpoint(output_file)

        try:
            with open(file, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"파일 로드 실패: {file} → {e}")
            continue

        pending = [item for item in data if item.get("essay_id") not in done_ids]
        logger.info(f"{file.name}: 전체 {len(data)}개 중 {len(pending)}개 처리 예정")

        all_results: list = []
        if output_file.exists() and done_ids:
            with open(output_file, encoding="utf-8") as f:
                all_results = json.load(f)

        for idx, item in enumerate(pending, start=1):
            essay_id = item.get("essay_id")
            logger.info(f"[{idx}/{len(pending)}] 처리 시작: {essay_id}")

            try:
                result = process_essay(item, essay_index, pipeline, config.domain_sleep)
                if result:
                    append_result(output_file, result, all_results)
                    logger.info(f"완료: {essay_id}")
                else:
                    logger.warning(f"결과 없음: {essay_id}")
            except Exception as e:
                logger.error(f"처리 오류 [{essay_id}]: {e}")

            time.sleep(1.0)

        logger.info(f"[DONE] {file.name} → {output_file}")


# =========================
# 진입점
# =========================
def main():
    config = Config()
    config.validate()

    essay_index = build_essay_index(Path(config.essay_data_dir))
    if not essay_index:
        logger.error("Essay index가 비어 있습니다. 경로를 확인하세요.")
        return

    api_client = EssayJudgeClient(config)
    pipeline = MADPipeline(api_client, agent_sleep=config.agent_sleep)

    for in_dir, out_dir in zip(config.input_dirs, config.output_dirs):
        process_folder(in_dir, out_dir, essay_index, pipeline, config)


if __name__ == "__main__":
    main()