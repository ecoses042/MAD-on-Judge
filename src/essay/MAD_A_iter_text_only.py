"""
MAD-A Text-only ablation 실험.

outline.md의 실험 3-B 조건에 맞춰, strict/lenient 에이전트가
상대방의 점수 숫자는 보지 않고 텍스트 피드백만 참조해 점수를 조정한다.

원본 MAD_A_iter.py의 프롬프트 형식과 출력 구조는 유지하되,
조정 라운드에서 공유되는 JSON에서 점수 관련 필드만 제거한다.

CLI:
  python src/MAD_A_iter_text_only.py --judge-model gpt --iterations 5
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from judge_prompt_utils import (
    LENIENT_JUDGE_ADJUST_SYSTEM,
    STRICT_JUDGE_ADJUST_SYSTEM,
    build_adjust_prompt,
    compute_overall,
)
from MAD_A_iter import (
    API_KEY,
    IterConfig,
    IterMADPipeline,
    JudgeClient,
    build_essay_index,
    logger,
    process_folder,
)


TEXT_FIELDS = ("rationale_for_score", "adjustment_notes")


@dataclass
class TextOnlyIterConfig(IterConfig):
    def output_dirs(self) -> list[str]:
        n = self.n_iterations
        return [
            f"judge_results/mad_a_text_only/{self.judge_model}/iter{n}/gemma",
            f"judge_results/mad_a_text_only/{self.judge_model}/iter{n}/qwen",
            f"judge_results/mad_a_text_only/{self.judge_model}/iter{n}/llama",
            f"judge_results/mad_a_text_only/{self.judge_model}/iter{n}/gpt",
        ]


def extract_text_feedback(other_result: dict) -> dict:
    shared = {
        key: other_result[key].strip()
        for key in TEXT_FIELDS
        if isinstance(other_result.get(key), str) and other_result[key].strip()
    }
    if shared:
        return shared
    return {"adjustment_notes": "상대 심사위원의 텍스트 피드백이 제공되지 않았다."}


class TextOnlyIterMADPipeline(IterMADPipeline):
    """Text-only 조정 프롬프트를 사용하되, 각 라운드의 strict/lenient 2-병렬 실행은 유지한다."""

    def run(self, sample: dict) -> Optional[dict]:
        """
        MAD_A_iter.py와 동일하게 매 라운드 strict/lenient 추론 2개를 병렬 실행한다.
        Text-only ablation에서는 조정 프롬프트에서 상대의 점수 필드만 제거하고,
        실행 스케줄은 base pipeline과 동일하게 유지한다.
        """
        return super().run(sample)

    def _strict_adjust(self, sample: dict, other_result: dict, round_num: int) -> Optional[dict]:
        ctx = f"{sample.get('essay_id')}_{sample.get('target_domain')}_r{round_num}_strict_text_only"
        user_prompt = build_adjust_prompt(
            sample,
            extract_text_feedback(other_result),
            reviewer_label="관대한 심사위원의 평가",
            instruction="위 평가를 참고하여 자신의 점수를 조정하라. 반드시 json으로만 응답하라.",
        )
        result = self.client.call_json(STRICT_JUDGE_ADJUST_SYSTEM, user_prompt, temperature=0.0, context=ctx)
        if result:
            result["overall_judge"] = compute_overall(result)
        return result

    def _lenient_adjust(self, sample: dict, other_result: dict, round_num: int) -> Optional[dict]:
        ctx = f"{sample.get('essay_id')}_{sample.get('target_domain')}_r{round_num}_lenient_text_only"
        user_prompt = build_adjust_prompt(
            sample,
            extract_text_feedback(other_result),
            reviewer_label="엄격한 심사위원의 평가",
            instruction="위 평가를 참고하여 자신의 점수를 조정하라. 반드시 json으로만 응답하라.",
        )
        result = self.client.call_json(LENIENT_JUDGE_ADJUST_SYSTEM, user_prompt, temperature=0.0, context=ctx)
        if result:
            result["overall_judge"] = compute_overall(result)
        return result


def main() -> None:
    import argparse

    model_map = {"gemma": 0, "qwen": 1, "llama": 2, "gpt": 3}
    judge_choices = ["gpt", "gemma", "qwen"]

    def run_for_iterations(iterations: int, model_name: Optional[str], judge_model: str) -> None:
        config = TextOnlyIterConfig(n_iterations=iterations, judge_model=judge_model)
        config.validate()

        essay_index = build_essay_index(Path(config.essay_data_dir))
        if not essay_index:
            logger.error("Essay index가 비어 있습니다.")
            return

        api_client = JudgeClient(config)
        logger.info(f"[INFO] Judge 모델: {api_client.model_name} (text-only)")
        pipeline = TextOnlyIterMADPipeline(
            api_client,
            n_iterations=config.n_iterations,
            agent_sleep=config.agent_sleep,
        )
        out_dirs = config.output_dirs()

        if model_name:
            idx = model_map[model_name]
            logger.info(
                f"단독 실행: {model_name}, iter={iterations} "
                f"({config.input_dirs[idx]} → {out_dirs[idx]})"
            )
            process_folder(config.input_dirs[idx], out_dirs[idx], essay_index, pipeline, config)
            return

        if judge_model == "qwen" or judge_model == "gemma":
            
            target_models = ["llama", "gpt"]
            logger.info(f"qwen judge: llama/gpt 에세이 모델만 실행 (iter={iterations}, text-only)")
        else:
            target_models = list(model_map.keys())
            logger.info(f"전체 순차 실행: iter={iterations} (text-only)")

        for name in target_models:
            idx = model_map[name]
            process_folder(config.input_dirs[idx], out_dirs[idx], essay_index, pipeline, config)

    parser = argparse.ArgumentParser(description="MAD-A Text-only ablation 실험")
    parser.add_argument(
        "--model",
        choices=["gemma", "qwen", "llama", "gpt"],
        default=None,
        help="처리할 에세이 모델 이름. 생략 시 전체 순차 실행.",
    )
    parser.add_argument(
        "--judge-model",
        choices=judge_choices,
        default=None,
        help="Judge로 사용할 모델. 생략 시 gpt→gemma→qwen 순서로 실행",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="교환 횟수. 생략하면 기본으로 5회를 실행",
    )
    args = parser.parse_args()

    if args.judge_model:
        judge_models = [args.judge_model]
    else:
        judge_models = judge_choices.copy()
        if not API_KEY and "gpt" in judge_models:
            judge_models.remove("gpt")
            logger.warning("OPENAI_API_KEY가 없어 기본 실행에서 gpt judge는 건너뜁니다.")

    iterations_list = [args.iterations] if args.iterations else [5]

    for judge_model in judge_models:
        for iterations in iterations_list:
            run_for_iterations(iterations, args.model, judge_model)


if __name__ == "__main__":
    main()
