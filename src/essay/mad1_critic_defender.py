"""
MAD (Multi-Agent Debate) 에세이 자동 평가 파이프라인
Critic → Defender → Final Judge 3단계 구조
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI, RateLimitError
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from env_utils import load_project_env

load_project_env(__file__)

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
    output_dirs: list[str] = field(default_factory=lambda: ["judge_results/mad1/gemma", "judge_results/mad1/qwen", "judge_results/mad1/llama", "judge_results/mad1/gpt"])

    max_retries: int = 3
    retry_base_wait: float = 5.0
    domain_sleep: float = 1.0      # 기존 0.3보다 넉넉하게
    agent_sleep: float = 0.7       # critic/defender/final 사이 간격 추가

    def validate(self):
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        if len(self.input_dirs) != len(self.output_dirs):
            raise ValueError("input_dirs와 output_dirs의 길이가 일치하지 않습니다.")

from prompts.essay import (
    CRITIC_SYSTEM,
    DEFENDER_SYSTEM,
    DOMAINS,
    FINAL_JUDGE_SYSTEM,
    build_user_prompt,
    compute_overall,
)

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
            "prompt_text": essay_data.get("prompt_text", ""),
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
                "content": 3,
                "organization": 3,
                "expression": 3,
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
