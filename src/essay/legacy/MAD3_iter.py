"""
MAD3_iter — Debate Iteration 증가 실험
MAD3의 2회 교환(strict_initial→lenient_initial→strict_adjusted→lenient_adjusted)을
N회 교환으로 확장한다.

출력 구조 (도메인별):
{
  "round_0": {"strict": {...}, "lenient": {...}},
  "round_1": {"strict": {...}, "lenient": {...}},
  ...
  "round_N-1": {"strict": {...}, "lenient": {...}},
  "final": {"4개 점수(float 평균)", "overall_judge"}
}

CLI:
  python src/MAD3_iter.py --model gemma --iterations 3
  python src/MAD3_iter.py --model gemma --iterations 5
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from openai import OpenAI, RateLimitError
from env_utils import load_project_env
from judge_prompt_utils import (
    DOMAINS,
    LENIENT_JUDGE_ADJUST_SYSTEM,
    LENIENT_JUDGE_SYSTEM,
    STRICT_JUDGE_ADJUST_SYSTEM,
    STRICT_JUDGE_SYSTEM,
    build_adjust_prompt,
    build_user_prompt,
    compute_overall,
)

load_project_env(__file__)
# MAD3의 공유 상수·함수 재사용
from MAD3 import (
    build_essay_index,
    load_checkpoint,
    append_result,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODELS = {
    "gemma": "google/gemma-3-4b",
    "qwen": "qwen/qwen3.5-9b",
}
API_KEY = os.environ.get("OPENAI_API_KEY", "")

# =========================
# 설정
# =========================
@dataclass
class IterConfig:
    api_key: str = os.environ.get("OPENAI_API_KEY", "")
    judge_model: str = "gpt"    # "gpt", "gemma", "qwen"
    model_name: str = "gpt-4o-mini"
    essay_data_dir: str = "data/selected_prompt_jsons"
    input_dirs: list[str] = field(default_factory=lambda: [
        "inference_results/gemma",
        "inference_results/qwen",
        "inference_results/llama",
        "inference_results/gpt",
    ])
    n_iterations: int = 3       # 총 교환 횟수 (2 = 기존 MAD3과 동일)
    max_retries: int = 3
    retry_base_wait: float = 5.0
    domain_sleep: float = 1.0
    agent_sleep: float = 0.7

    def __post_init__(self):
        if self.judge_model in LM_STUDIO_MODELS:
            self.agent_sleep = 0.0
            self.domain_sleep = 0.0

    def output_dirs(self) -> list[str]:
        n = self.n_iterations
        return [
            f"judge_results/exp05_iter/{self.judge_model}/iter{n}/gemma",
            f"judge_results/exp05_iter/{self.judge_model}/iter{n}/qwen",
            f"judge_results/exp05_iter/{self.judge_model}/iter{n}/llama",
            f"judge_results/exp05_iter/{self.judge_model}/iter{n}/gpt",
        ]

    def validate(self):
        if self.judge_model not in LM_STUDIO_MODELS and not self.api_key:
            raise ValueError("api_key가 설정되지 않았습니다.")
        if self.n_iterations < 2:
            raise ValueError("n_iterations는 최소 2 이상이어야 합니다.")


# =========================
# API 클라이언트
# =========================
class JudgeClient:
    def __init__(self, config: IterConfig):
        self.config = config
        if config.judge_model in LM_STUDIO_MODELS:
            self.client = OpenAI(api_key="lm-studio", base_url=LM_STUDIO_BASE_URL)
            self.model_name = LM_STUDIO_MODELS[config.judge_model]
        else:
            self.client = OpenAI(api_key=config.api_key)
            self.model_name = config.model_name
        self.use_json_format = self.model_name.startswith("gpt-")
        self._debug_dir = Path("debug/raw_responses") / config.judge_model
        self._debug_dir.mkdir(parents=True, exist_ok=True)

    def _save_raw_response(self, raw: str, context: str) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_ctx = context.replace("/", "_").replace(" ", "_")
        path = self._debug_dir / f"{ts}_{safe_ctx}.json"
        try:
            path.write_text(
                json.dumps({"context": context, "raw_response": raw}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.debug(f"raw 응답 저장: {path}")
        except OSError as e:
            logger.warning(f"raw 응답 저장 실패: {e}")

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """```json ... ``` 또는 ``` ... ``` 마크다운 코드블록을 제거한다."""
        stripped = text.strip()
        if stripped.startswith("```"):
            # 첫 줄(```json 또는 ```) 제거
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1:]
            # 끝의 ``` 제거
            if stripped.endswith("```"):
                stripped = stripped[: stripped.rfind("```")]
        return stripped.strip()

    def call_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0,
                  context: str = "unknown") -> Optional[dict]:
        for attempt in range(self.config.max_retries):
            raw_content: Optional[str] = None
            try:
                create_kwargs = dict(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=1024,
                )
                if self.use_json_format:
                    create_kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**create_kwargs)
                raw_content = response.choices[0].message.content
                return json.loads(self._strip_code_fence(raw_content))
            except RateLimitError:
                wait = self.config.retry_base_wait * (attempt + 1)
                logger.warning(f"RateLimit: {wait:.0f}s 대기 (시도 {attempt + 1}/{self.config.max_retries})")
                time.sleep(wait)
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 실패 [{context}]: {e}")
                logger.error(f"raw 응답 (앞 300자): {(raw_content or '')[:300]}")
                self._save_raw_response(raw_content or "", context)
                return None
            except Exception as e:
                logger.error(f"API 호출 오류 [{context}]: {e}")
                return None
        logger.error("최대 재시도 초과")
        return None


# =========================
# 반복 MAD 파이프라인
# =========================
class IterMADPipeline:
    """
    N회 교환 구조:
      round_0: strict_r0(temp=0.1), lenient_r0(temp=0.1) — 독립 평가
      round_1: strict_r1(lenient_r0 참조), lenient_r1(strict_r0 참조) — (temp=0.0)
      ...
      round_N-1: strict_rN-1(lenient_rN-2 참조), lenient_rN-1(strict_rN-2 참조)
    final = avg(strict_rN-1, lenient_rN-1)
    """

    def __init__(self, client: JudgeClient, n_iterations: int = 3, agent_sleep: float = 0.7):
        self.client = client
        self.n_iterations = n_iterations
        self.agent_sleep = agent_sleep

    def _strict_initial(self, sample: dict) -> Optional[dict]:
        ctx = f"{sample.get('essay_id')}_{sample.get('target_domain')}_r0_strict"
        result = self.client.call_json(
            STRICT_JUDGE_SYSTEM,
            build_user_prompt(sample) + "\n\n반드시 json으로만 응답하라.",
            temperature=0.1,
            context=ctx,
        )
        if result:
            result["overall_judge"] = compute_overall(result)
        return result

    def _lenient_initial(self, sample: dict) -> Optional[dict]:
        ctx = f"{sample.get('essay_id')}_{sample.get('target_domain')}_r0_lenient"
        result = self.client.call_json(
            LENIENT_JUDGE_SYSTEM,
            build_user_prompt(sample) + "\n\n반드시 json으로만 응답하라.",
            temperature=0.1,
            context=ctx,
        )
        if result:
            result["overall_judge"] = compute_overall(result)
        return result

    def _strict_adjust(self, sample: dict, other_result: dict, round_num: int) -> Optional[dict]:
        ctx = f"{sample.get('essay_id')}_{sample.get('target_domain')}_r{round_num}_strict"
        user_prompt = build_adjust_prompt(
            sample,
            other_result,
            reviewer_label="관대한 심사위원의 평가",
            instruction="위 평가를 참고하여 자신의 점수를 조정하라. 반드시 json으로만 응답하라.",
        )
        result = self.client.call_json(STRICT_JUDGE_ADJUST_SYSTEM, user_prompt, temperature=0.0, context=ctx)
        if result:
            result["overall_judge"] = compute_overall(result)
        return result

    def _lenient_adjust(self, sample: dict, other_result: dict, round_num: int) -> Optional[dict]:
        ctx = f"{sample.get('essay_id')}_{sample.get('target_domain')}_r{round_num}_lenient"
        user_prompt = build_adjust_prompt(
            sample,
            other_result,
            reviewer_label="엄격한 심사위원의 평가",
            instruction="위 평가를 참고하여 자신의 점수를 조정하라. 반드시 json으로만 응답하라.",
        )
        result = self.client.call_json(LENIENT_JUDGE_ADJUST_SYSTEM, user_prompt, temperature=0.0, context=ctx)
        if result:
            result["overall_judge"] = compute_overall(result)
        return result

    def run(self, sample: dict) -> Optional[dict]:
        rounds: list[dict] = []

        # round_0: 독립 평가
        strict_r = self._strict_initial(sample)
        if not strict_r:
            logger.warning(f"round_0 strict 실패: {sample.get('essay_id')}, {sample.get('target_domain')}")
            return None
        time.sleep(self.agent_sleep)

        lenient_r = self._lenient_initial(sample)
        if not lenient_r:
            logger.warning(f"round_0 lenient 실패: {sample.get('essay_id')}, {sample.get('target_domain')}")
            return None
        time.sleep(self.agent_sleep)

        rounds.append({"strict": strict_r, "lenient": lenient_r})

        # round_1 ~ round_N-1: 상호 참조 조정
        for r in range(1, self.n_iterations):
            prev_strict = rounds[r - 1]["strict"]
            prev_lenient = rounds[r - 1]["lenient"]

            new_strict = self._strict_adjust(sample, prev_lenient, round_num=r)
            if not new_strict:
                logger.warning(f"round_{r} strict 실패: {sample.get('essay_id')}, {sample.get('target_domain')}")
                return None
            time.sleep(self.agent_sleep)

            new_lenient = self._lenient_adjust(sample, prev_strict, round_num=r)
            if not new_lenient:
                logger.warning(f"round_{r} lenient 실패: {sample.get('essay_id')}, {sample.get('target_domain')}")
                return None
            time.sleep(self.agent_sleep)

            rounds.append({"strict": new_strict, "lenient": new_lenient})

        # final: 마지막 라운드 평균
        keys = ["domain_match", "score_rationale_consistency", "specificity", "groundedness"]
        last = rounds[-1]
        final = {
            k: round((last["strict"].get(k, 3) + last["lenient"].get(k, 3)) / 2, 1)
            for k in keys
        }
        final["overall_judge"] = round(
            (last["strict"]["overall_judge"] + last["lenient"]["overall_judge"]) / 2, 1
        )

        result = {f"round_{i}": rounds[i] for i in range(len(rounds))}
        result["final"] = final
        return result


# =========================
# 단일 Essay 처리
# =========================
def process_essay(item: dict, essay_index: dict, pipeline: IterMADPipeline, sleep_sec: float) -> Optional[dict]:
    essay_id = item.get("essay_id")
    essay_data = essay_index.get(essay_id)

    if not essay_data:
        logger.warning(f"Essay 데이터 없음: {essay_id}")
        return None

    result_item = {
        "essay_id": essay_id,
        "status": "ok",
        "prediction": item.get("prediction", {}),
        "judge": {},
    }

    fallback_round = {
        "strict": {"domain_match": 3, "score_rationale_consistency": 3, "specificity": 3, "groundedness": 3, "overall_judge": 3.0, "rationale_for_score": "처리 실패"},
        "lenient": {"domain_match": 3, "score_rationale_consistency": 3, "specificity": 3, "groundedness": 3, "overall_judge": 3.0, "rationale_for_score": "처리 실패"},
    }
    fallback = {f"round_{i}": fallback_round for i in range(pipeline.n_iterations)}
    fallback["final"] = {"domain_match": 3.0, "score_rationale_consistency": 3.0, "specificity": 3.0, "groundedness": 3.0, "overall_judge": 3.0}

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
        domain_result = pipeline.run(sample)

        result_item["judge"][domain] = domain_result if domain_result else fallback
        time.sleep(sleep_sec)

    if not result_item["judge"]:
        result_item["status"] = "fail"

    return result_item


# =========================
# 폴더 단위 처리
# =========================
def process_folder(input_folder: str, output_folder: str, essay_index: dict, pipeline: IterMADPipeline, config: IterConfig):
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
            time.sleep(0.0 if config.judge_model in LM_STUDIO_MODELS else 1.0)

        logger.info(f"[DONE] {file.name} → {output_file}")


# =========================
# 진입점
# =========================
def main():
    import argparse

    MODEL_MAP = {"gemma": 0, "qwen": 1, "llama": 2, "gpt": 3}
    JUDGE_CHOICES = ["gpt", "gemma", "qwen"]

    def run_for_iterations(iterations: int, model_name: Optional[str], judge_model: str):
        config = IterConfig(n_iterations=iterations, judge_model=judge_model)
        config.validate()

        essay_index = build_essay_index(Path(config.essay_data_dir))
        if not essay_index:
            logger.error("Essay index媛 鍮꾩뼱 ?덉뒿?덈떎.")
            return

        api_client = JudgeClient(config)
        logger.info(f"[INFO] Judge 모델: {api_client.model_name}")
        pipeline = IterMADPipeline(api_client, n_iterations=config.n_iterations, agent_sleep=config.agent_sleep)
        out_dirs = config.output_dirs()

        if model_name:
            idx = MODEL_MAP[model_name]
            logger.info(f"단독 실행: {model_name}, iter={iterations} ({config.input_dirs[idx]} → {out_dirs[idx]})")
            process_folder(config.input_dirs[idx], out_dirs[idx], essay_index, pipeline, config)
        else:
            if judge_model == "qwen":
                target_models = ["llama", "gpt"]
                logger.info(f"qwen judge: llama/gpt 에세이 모델만 실행 (iter={iterations})")
            else:
                target_models = list(MODEL_MAP.keys())
                logger.info(f"전체 순차 실행: iter={iterations}")
            for name in target_models:
                idx = MODEL_MAP[name]
                process_folder(config.input_dirs[idx], out_dirs[idx], essay_index, pipeline, config)

    parser = argparse.ArgumentParser(description="MAD3 Iteration 증가 실험")
    parser.add_argument("--model", choices=list(MODEL_MAP.keys()), default=None,
                        help="처리할 에세이 모델 이름. 생략 시 전체 순차 실행.")
    parser.add_argument("--judge-model", choices=JUDGE_CHOICES, default=None,
                        help="Judge로 사용할 모델: gpt(GPT-4o-mini), gemma(LM Studio google/gemma-3-4b), qwen(LM Studio qwen/qwen3.5-9b). 생략 시 gpt→gemma→qwen 순서로 실행")
    parser.add_argument("--iterations", type=int, default=None,
                        help="교환 횟수. 생략하면 기본으로 3회 후 5회를 순서대로 실행")
    args = parser.parse_args()

    if args.judge_model:
        judge_models = [args.judge_model]
    else:
        judge_models = JUDGE_CHOICES.copy()
        if not API_KEY and "gpt" in judge_models:
            judge_models.remove("gpt")
            logger.warning("OPENAI_API_KEY가 없어 기본 실행에서 gpt judge는 건너뜁니다.")
    iterations_list = [args.iterations] if args.iterations else [3, 5]

    for judge_model in judge_models:
        for iterations in iterations_list:
            run_for_iterations(iterations, args.model, judge_model)


if __name__ == "__main__":
    main()
