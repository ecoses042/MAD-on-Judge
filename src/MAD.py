"""
MAD (Multi-Agent Debate) 에세이 자동 평가 파이프라인
Critic → Defender → Final Judge 3단계 구조
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
# 설정 (환경변수 우선, 폴백은 기본값)
# =========================
@dataclass
class Config:
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model_name: str = "gpt-4o-mini"
    essay_data_dir: str = "data/selected_prompt_jsons"
    input_dirs: list[str] = field(default_factory=lambda: ["inference_results/gemma", "inference_results/qwen", "inference_results/llama", "inference_results/gpt"])
    output_dirs: list[str] = field(default_factory=lambda: ["judge_results/exp02_mad/gemma", "judge_results/exp02_mad/qwen", "judge_results/exp02_mad/llama", "judge_results/exp02_mad/gpt"])

    max_retries: int = 3
    retry_base_wait: float = 5.0
    domain_sleep: float = 1.0      # 기존 0.3보다 넉넉하게
    agent_sleep: float = 0.7       # critic/defender/final 사이 간격 추가

    def validate(self):
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        if len(self.input_dirs) != len(self.output_dirs):
            raise ValueError("input_dirs와 output_dirs의 길이가 일치하지 않습니다.")

# =========================
# 시스템 프롬프트 (상수 분리)
# =========================
COMMON_CRITERIA = """
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

""".strip()
CRITIC_SYSTEM = f"""
[역할]
너는 매우 공격적인 오류 탐지 심사위원이다.

목표:
- rationale의 취약점을 최대한 많이 찾아라

{COMMON_CRITERIA}

---

[판단 규칙]

- 반드시 최소 2개 이상의 문제를 찾아라
- 문제가 없으면 specificity 부족으로 간주하라
- 애매하면 감점하라

---

[강제 감점 조건]

다음은 반드시 major issue:

- groundedness 위반 (없는 내용)
- domain mismatch
- 핵심 근거 부족
- generic 설명

---

[출력 형식]

{{
  "major_issues": [
    {{
      "criterion": "groundedness",
      "reason": "essay에 없는 내용을 근거로 사용"
    }}
  ],
  "minor_issues": [
    {{
      "criterion": "specificity",
      "reason": "구체적 근거 부족"
    }}
  ],
  "provisional_score": 1~5
}}
""".strip()

DEFENDER_SYSTEM = f"""
[역할]
너는 근거를 최대한 공정하게 방어하는 평가자이다.

목표:
- rationale이 왜 타당한지 설명하라

{COMMON_CRITERIA}

---

[판단 규칙]

- 반드시 최소 1개 이상의 강점을 제시하라
- 강점은 반드시 essay_text 기반
- 억지 방어 금지

---

[강점 정의]

- 실제 문장/논지 기반
- generic 표현 금지

---

[출력 형식]

{{
  "strengths": [
    {{
      "criterion": "domain_match",
      "reason": "논지 전개 기준으로 평가"
    }}
  ],
  "remaining_concerns": [
    {{
      "criterion": "specificity",
      "reason": "구체성 부족"
    }}
  ],
  "provisional_score": 1~5
}}
""".strip()

FINAL_JUDGE_SYSTEM = """
[역할]
당신은 한국어 에세이 채점 결과의 타당성을 최종 판정하는 공정한 중재자이다.

당신은 아래 두 평가 결과를 참고한다:
- Critic: 오류 중심 평가
- Defender: 정당화 중심 평가

당신의 임무는 Critic과 Defender 중 어느 쪽이 essay_text에 더 직접적이고 설득력 있게 근거하고 있는지를 비교하여, 최종 점수를 공정하게 결정하는 것이다.

[평가 대상]
모델이 특정 영역(target_domain)에 대해 제시한:
1) predicted_score
2) rationale

그리고 Critic / Defender의 평가 결과

[핵심 원칙]
- 1~5 전 구간을 적극적으로 사용하라.
- Critic의 지적이 존재한다는 이유만으로 자동 감점하지 마라.
- Defender의 방어가 essay_text에 직접 근거하면 반드시 동등하게 반영하라.
- 최종 판단은 "문제의 개수"가 아니라 "더 설득력 있는 근거가 어느 쪽에 있는가"로 결정하라.
- 감점 조건뿐 아니라 고득점 조건도 반드시 검토하라.
- generic한 총평, 상투적 표현, 템플릿형 설명은 낮게 평가하라.
- essay_text에 없는 내용을 암시하거나 만들어내면 groundedness는 반드시 1~2점이다.
- target_domain과 다른 영역 기준을 섞으면 domain_match는 반드시 1~2점이다.

[평가 항목]
1. domain_match
- rationale이 target_domain의 평가 기준에 맞는 근거를 제시하는가

2. score_rationale_consistency
- predicted_score와 rationale의 내용이 서로 잘 맞는가

3. specificity
- rationale이 구체적인가
- 실제 글의 특정 문장, 표현, 논지, 문단 전개, 오류 양상을 짚는가

4. groundedness
- rationale이 실제 essay_text에 근거하는가
- 에세이에 없는 내용을 만들어내지 않는가

[target_domain별 판단 기준]
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

[강제 감점 규칙]
다음 중 하나라도 해당하면 관련 항목은 2점 이하를 적극 검토하라.
- target_domain과 무관한 기준으로 설명함
- essay_text에 없는 내용을 근거처럼 제시함
- predicted_score를 정당화할 핵심 증거가 없음
- 지나치게 일반적이고 템플릿 같은 설명만 함
- 실제 글의 장점/결함을 확인 가능한 수준으로 짚지 못함

[고득점 부여 규칙]
다음 조건이 충족되면 4점 이상을 적극 검토하라.
- rationale이 해당 domain 기준을 정확히 적용함
- rationale이 essay_text의 실제 내용과 직접 연결됨
- rationale이 generic하지 않고 구체적임
- predicted_score를 설득력 있게 정당화함
- Defender의 방어가 Critic의 지적보다 essay_text에 더 직접적으로 근거함

[점수 기준]
5점:
- 매우 강한 타당성
- Critic의 핵심 지적이 대부분 반박되며, rationale이 구체적이고 정확하고 essay_text와 긴밀히 연결됨

4점:
- 전반적으로 타당함
- 일부 약점은 있으나 Defender의 방어가 더 설득력 있고, rationale이 충분히 근거 있고 구체적임

3점:
- Critic과 Defender가 모두 일정 부분 타당하며 우열이 뚜렷하지 않음
- 단, 3점은 정말 균형이 팽팽한 경우에만 사용하라

2점:
- 문제점이 뚜렷함
- Critic의 지적이 Defender의 방어보다 더 설득력 있으며, 영역 혼동, 근거 부족, 일반론적 설명, essay와의 연결 부족이 보임

1점:
- 거의 타당하지 않음
- Critic의 지적이 명백하고 Defender가 이를 반박하지 못함
- 명백한 영역 혼동, 환각, 근거 부재, 점수-설명 모순이 있음

[판정 절차]
각 항목(domain_match, score_rationale_consistency, specificity, groundedness)에 대해 반드시 다음 순서로 판단하라:
1. Critic의 핵심 주장 요약
2. Defender의 핵심 주장 요약
3. 둘 중 essay_text에 더 직접적으로 근거한 쪽 선택
4. 그 결과를 바탕으로 점수 부여

[출력 규칙]
- 설명 문장은 출력하지 말고 JSON 객체 하나만 출력하라.
- 코드블록 마크다운을 사용하지 마라.
- 모든 점수는 반드시 1~5의 정수여야 한다.

[출력 형식]
{
  "domain_match": 1,
  "score_rationale_consistency": 1,
  "specificity": 1,
  "groundedness": 1,
  "winner_side": {
    "domain_match": "critic|defender|tie",
    "score_rationale_consistency": "critic|defender|tie",
    "specificity": "critic|defender|tie",
    "groundedness": "critic|defender|tie"
  }
}

이제 아래 정보를 바탕으로 평가하라.
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
        """JSON 응답을 반환하는 API 호출. 실패 시 None 반환."""
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

        logger.error(f"최대 재시도 초과")
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

    # 최솟값에 따른 상한 적용
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

    def defender(self, sample: dict) -> Optional[dict]:
        return self.client.call_json(
            DEFENDER_SYSTEM,
            build_user_prompt(sample) + "\n\n반드시 json으로만 응답하라.",
            temperature=0.1,
        )

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

    def run(self, sample: dict) -> Optional[dict]:
        critic_result = self.critic(sample)
        if not critic_result:
            logger.warning(
                f"Critic 실패: essay_id={sample.get('essay_id', '?')}, domain={sample.get('target_domain')}"
            )
            return None

        time.sleep(self.agent_sleep)

        defender_result = self.defender(sample)
        if not defender_result:
            logger.warning(
                f"Defender 실패: essay_id={sample.get('essay_id', '?')}, domain={sample.get('target_domain')}"
            )
            return None

        time.sleep(self.agent_sleep)

        final_result = self.final_judge(sample, critic_result, defender_result)
        if not final_result:
            logger.warning(
                f"Final Judge 실패: essay_id={sample.get('essay_id', '?')}, domain={sample.get('target_domain')}"
            )
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
    """이미 처리된 essay_id 목록을 반환."""
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
    """결과를 즉시 파일에 저장 (중간 실패 대비)."""
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