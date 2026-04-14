"""
MAD3 (Multi-Agent Debate) 에세이 자동 평가 파이프라인
Strict Judge + Lenient Judge → 상호 참조 후 조정 → 평균 합산
[구조] 1단계 독립 평가 → 2단계 상호 참조 조정 → 3단계 평균 합산
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
    output_dirs: list[str] = field(default_factory=lambda: ["judge_results/exp04_mad3/gemma", "judge_results/exp04_mad3/qwen", "judge_results/exp04_mad3/llama", "judge_results/exp04_mad3/gpt"])
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
     - rationale이 애매하게 맞고 애매하게 틀린 경우 (단, 이 경우 반드시 근거 명시)
2점: 주요 결함이 1개 이상 존재하고 rationale이 predicted_score를 정당화하지 못함
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


STRICT_JUDGE_SYSTEM = f"""
[역할]
너는 rationale의 품질을 극도로 엄격하게 평가하는 심사위원이다.
평가 대상은 오직 rationale이다 — essay_text 자체의 품질을 평가하지 마라.

[기본 태도]
- 의심하라. 모든 rationale은 결함이 있다고 전제하고 시작하라.
- "논리적이다", "잘 썼다", "전반적으로 좋다" 수준의 표현은 근거로 인정하지 않는다.
- 영역(domain) 혼동, 점수-설명 모순, essay_text에 없는 내용 인용은 즉시 1~2점 처리한다.
- 구체적 문장이나 표현 인용이 rationale에 없으면 specificity와 groundedness를 각각 2점 이하로 처리한다.
- predicted_score가 4~5점인데 rationale에 직접 인용 가능한 구체적 강점이 2개 미만이면
  score_rationale_consistency를 2점으로 처리한다.

{COMMON_CRITERIA}

[출력 규칙]
- JSON 객체 하나만 출력하라. 코드블록 마크다운 사용 금지.
- 모든 점수는 1~5 정수.
- rationale_for_score는 한국어로, 각 항목별 감점 근거를 명시하라.

[출력 형식]
{{
  "domain_match": 1~5,
  "score_rationale_consistency": 1~5,
  "specificity": 1~5,
  "groundedness": 1~5,
  "overall_judge": 소수점 1자리 실수,
  "rationale_for_score": "각 항목별 감점 이유를 포함한 종합 평가"
}}
""".strip()


LENIENT_JUDGE_SYSTEM = f"""
[역할]
너는 rationale의 품질을 최대한 관대하게 평가하는 심사위원이다.
평가 대상은 오직 rationale이다 — essay_text 자체의 품질을 평가하지 마라.

[기본 태도]
- 가능성을 찾아라. rationale에서 긍정적으로 해석할 수 있는 여지가 조금이라도 있으면 높은 점수를 부여하라.
- essay_text에 근거가 될 수 있는 내용이 존재하고, rationale이 그 방향을 가리키고 있다면
  specificity와 groundedness를 높게 인정하라.
- 표현이 다소 일반적이더라도, 평가 방향(긍정/부정)이 essay_text와 일치하면 점수를 높게 부여하라.
- 점수-설명이 대략적으로라도 방향이 일치한다면 score_rationale_consistency를 최소 3점으로 처리하라.
- 명백한 hallucination(essay_text에 전혀 존재하지 않는 내용을 인용)이 아니면 감점하지 마라.

{COMMON_CRITERIA}

[출력 규칙]
- JSON 객체 하나만 출력하라. 코드블록 마크다운 사용 금지.
- 모든 점수는 1~5 정수.
- rationale_for_score는 한국어로, 각 항목별 가점 근거를 명시하라.

[출력 형식]
{{
  "domain_match": 1~5,
  "score_rationale_consistency": 1~5,
  "specificity": 1~5,
  "groundedness": 1~5,
  "overall_judge": 소수점 1자리 실수,
  "rationale_for_score": "각 항목별 인정한 근거를 포함한 종합 평가"
}}
""".strip()


STRICT_JUDGE_ADJUST_SYSTEM = f"""
[역할]
너는 초기 평가를 마친 엄격한 심사위원이다.
지금 너는 관대한 심사위원의 초기 평가 결과를 보고, 자신의 점수를 재검토해야 한다.

[조정 원칙]
- 이것은 토론이 아니다. 이기려 하거나 반박하려 하지 마라. 목적은 점수 조정이다.
- 관대한 심사위원이 인정한 근거 중, rationale 텍스트에 실제로 존재하는 내용은 인정하라.
- 네가 지나치게 가혹하게 처리했거나 놓쳤던 항목은 점수를 올려라.
- 단, 관대한 심사위원이 essay_text에서 새로 발굴한 근거로 rationale을 방어한 경우는 인정하지 않는다.
  — rationale 텍스트에 이미 있는 내용만 인정 가능하다.
- 확신이 있는 항목은 기존 점수를 유지해도 된다.

{COMMON_CRITERIA}

[출력 규칙]
- JSON 객체 하나만 출력하라. 코드블록 마크다운 사용 금지.
- 모든 점수는 1~5 정수.
- adjustment_notes는 한국어로, 변경한 항목과 이유, 유지한 항목과 이유를 명시하라.

[출력 형식]
{{
  "domain_match": 1~5,
  "score_rationale_consistency": 1~5,
  "specificity": 1~5,
  "groundedness": 1~5,
  "overall_judge": 소수점 1자리 실수,
  "adjustment_notes": "항목별 조정 내역 및 사유"
}}
""".strip()


LENIENT_JUDGE_ADJUST_SYSTEM = f"""
[역할]
너는 초기 평가를 마친 관대한 심사위원이다.
지금 너는 엄격한 심사위원의 초기 평가 결과를 보고, 자신의 점수를 재검토해야 한다.

[조정 원칙]
- 이것은 토론이 아니다. 이기려 하거나 반박하려 하지 마라. 목적은 점수 조정이다.
- 엄격한 심사위원이 지적한 결함 중, rationale에서 실제로 확인되는 것은 인정하라.
- 네가 지나치게 관대하게 처리했거나 근거 없이 높게 준 항목은 점수를 내려라.
- 단, 엄격한 심사위원이 지나치게 가혹하게 처리한 항목에 대해서는 기존 점수를 유지해도 된다.
- 확신이 있는 항목은 기존 점수를 유지해도 된다.

{COMMON_CRITERIA}

[출력 규칙]
- JSON 객체 하나만 출력하라. 코드블록 마크다운 사용 금지.
- 모든 점수는 1~5 정수.
- adjustment_notes는 한국어로, 변경한 항목과 이유, 유지한 항목과 이유를 명시하라.

[출력 형식]
{{
  "domain_match": 1~5,
  "score_rationale_consistency": 1~5,
  "specificity": 1~5,
  "groundedness": 1~5,
  "overall_judge": 소수점 1자리 실수,
  "adjustment_notes": "항목별 조정 내역 및 사유"
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
                    max_tokens=1024,
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

    def strict_judge(self, sample: dict) -> Optional[dict]:
        result = self.client.call_json(
            STRICT_JUDGE_SYSTEM,
            build_user_prompt(sample) + "\n\n반드시 json으로만 응답하라.",
            temperature=0.1,
        )
        if result:
            result["overall_judge"] = compute_overall(result)
        return result

    def lenient_judge(self, sample: dict) -> Optional[dict]:
        result = self.client.call_json(
            LENIENT_JUDGE_SYSTEM,
            build_user_prompt(sample) + "\n\n반드시 json으로만 응답하라.",
            temperature=0.1,
        )
        if result:
            result["overall_judge"] = compute_overall(result)
        return result

    def strict_adjust(self, sample: dict, lenient_initial: dict) -> Optional[dict]:
        user_prompt = (
            build_user_prompt(sample)
            + f"\n\n[관대한 심사위원의 초기 평가]\n{json.dumps(lenient_initial, ensure_ascii=False)}"
            + "\n\n위 평가를 참고하여 자신의 초기 점수를 조정하라. 반드시 json으로만 응답하라."
        )
        result = self.client.call_json(STRICT_JUDGE_ADJUST_SYSTEM, user_prompt, temperature=0.0)
        if result:
            result["overall_judge"] = compute_overall(result)
        return result

    def lenient_adjust(self, sample: dict, strict_initial: dict) -> Optional[dict]:
        user_prompt = (
            build_user_prompt(sample)
            + f"\n\n[엄격한 심사위원의 초기 평가]\n{json.dumps(strict_initial, ensure_ascii=False)}"
            + "\n\n위 평가를 참고하여 자신의 초기 점수를 조정하라. 반드시 json으로만 응답하라."
        )
        result = self.client.call_json(LENIENT_JUDGE_ADJUST_SYSTEM, user_prompt, temperature=0.0)
        if result:
            result["overall_judge"] = compute_overall(result)
        return result

    def run(self, sample: dict) -> Optional[dict]:
        # 1단계: 독립 평가
        strict_initial = self.strict_judge(sample)
        if not strict_initial:
            logger.warning(f"Strict Judge 실패: essay_id={sample.get('essay_id', '?')}, domain={sample.get('target_domain')}")
            return None

        time.sleep(self.agent_sleep)

        lenient_initial = self.lenient_judge(sample)
        if not lenient_initial:
            logger.warning(f"Lenient Judge 실패: essay_id={sample.get('essay_id', '?')}, domain={sample.get('target_domain')}")
            return None

        time.sleep(self.agent_sleep)

        # 2단계: 상호 참조 후 조정 (각자 상대방의 초기 결과를 봄)
        strict_adjusted = self.strict_adjust(sample, lenient_initial)
        if not strict_adjusted:
            logger.warning(f"Strict Adjust 실패: essay_id={sample.get('essay_id', '?')}, domain={sample.get('target_domain')}")
            return None

        time.sleep(self.agent_sleep)

        lenient_adjusted = self.lenient_adjust(sample, strict_initial)
        if not lenient_adjusted:
            logger.warning(f"Lenient Adjust 실패: essay_id={sample.get('essay_id', '?')}, domain={sample.get('target_domain')}")
            return None

        # 3단계: 최종 합의 (평균)
        keys = ["domain_match", "score_rationale_consistency", "specificity", "groundedness"]
        final = {
            k: round((strict_adjusted.get(k, 3) + lenient_adjusted.get(k, 3)) / 2, 1)
            for k in keys
        }
        final["overall_judge"] = round(
            (strict_adjusted["overall_judge"] + lenient_adjusted["overall_judge"]) / 2, 1
        )

        return {
            "strict_initial": strict_initial,
            "lenient_initial": lenient_initial,
            "strict_adjusted": strict_adjusted,
            "lenient_adjusted": lenient_adjusted,
            "final": final,
        }


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

    fallback_scores = {
        "domain_match": 3,
        "score_rationale_consistency": 3,
        "specificity": 3,
        "groundedness": 3,
        "overall_judge": 3.0,
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
                "strict_initial": {**fallback_scores, "rationale_for_score": "처리 실패"},
                "lenient_initial": {**fallback_scores, "rationale_for_score": "처리 실패"},
                "strict_adjusted": {**fallback_scores, "adjustment_notes": "처리 실패"},
                "lenient_adjusted": {**fallback_scores, "adjustment_notes": "처리 실패"},
                "final": {
                    "domain_match": 3.0,
                    "score_rationale_consistency": 3.0,
                    "specificity": 3.0,
                    "groundedness": 3.0,
                    "overall_judge": 3.0,
                },
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
    import argparse

    MODEL_MAP = {
        "gemma": 0,
        "qwen":  1,
        "llama": 2,
        "gpt":   3,
    }

    parser = argparse.ArgumentParser(description="MAD3 에세이 자동 평가")
    parser.add_argument(
        "--model",
        choices=list(MODEL_MAP.keys()),
        default=None,
        help="처리할 모델 이름 (gemma/qwen/llama/gpt). 생략 시 전체 순차 실행.",
    )
    args = parser.parse_args()

    config = Config()
    config.validate()

    essay_index = build_essay_index(Path(config.essay_data_dir))
    if not essay_index:
        logger.error("Essay index가 비어 있습니다. 경로를 확인하세요.")
        return

    api_client = EssayJudgeClient(config)
    pipeline = MADPipeline(api_client, agent_sleep=config.agent_sleep)

    if args.model:
        idx = MODEL_MAP[args.model]
        in_dir  = config.input_dirs[idx]
        out_dir = config.output_dirs[idx]
        logger.info(f"단독 실행 모드: {args.model} ({in_dir} → {out_dir})")
        process_folder(in_dir, out_dir, essay_index, pipeline, config)
    else:
        for in_dir, out_dir in zip(config.input_dirs, config.output_dirs):
            process_folder(in_dir, out_dir, essay_index, pipeline, config)


if __name__ == "__main__":
    main()
