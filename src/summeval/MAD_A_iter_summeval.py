import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # adds src/ to path
from env_utils import load_project_env

load_project_env(__file__)

import argparse
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass

from openai import OpenAI, RateLimitError

from summeval.summeval_prompt_utils import (
    LENIENT_JUDGE_ADJUST_SYSTEM,
    LENIENT_JUDGE_SYSTEM,
    SCORE_KEYS,
    STRICT_JUDGE_ADJUST_SYSTEM,
    STRICT_JUDGE_SYSTEM,
    build_adjust_prompt,
    build_user_prompt,
    compute_overall,
)

API_KEY = os.environ.get("OPENAI_API_KEY", "")
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODELS = {
    "gemma": "google/gemma-3-4b",
    "qwen": "qwen/qwen3.5-9b",
}
INPUT_BASE = Path("summeval_judge_input")
OUTPUT_BASE = Path("summeval_judge_results/mad_a_iter")


@dataclass
class IterConfig:
    judge_model: str = "gpt"
    n_iterations: int = 3
    max_retries: int = 3
    retry_base_wait: float = 5.0

    def validate(self) -> None:
        if self.n_iterations < 2:
            raise ValueError("n_iterations must be >= 2")
        if self.judge_model == "gpt" and not API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")


class JudgeClient:
    def __init__(self, config: IterConfig):
        self.config = config
        if config.judge_model in LM_STUDIO_MODELS:
            self.client = OpenAI(api_key="lm-studio", base_url=LM_STUDIO_BASE_URL)
            self.model_name = LM_STUDIO_MODELS[config.judge_model]
        else:
            self.client = OpenAI(api_key=API_KEY)
            self.model_name = "gpt-4o-mini"
        self.use_json_format = self.model_name.startswith("gpt-")

    @staticmethod
    def _extract_json_object(raw_text: str) -> dict:
        if not raw_text:
            raise json.JSONDecodeError("Empty response", raw_text, 0)

        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        start = cleaned.find("{")
        if start == -1:
            raise json.JSONDecodeError("No JSON object start found", cleaned, 0)

        depth = 0
        in_string = False
        escape = False
        end = None
        for idx in range(start, len(cleaned)):
            ch = cleaned[idx]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        if end is None:
            raise json.JSONDecodeError("No complete JSON object found", cleaned, start)
        return json.loads(cleaned[start:end])

    def call_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict | None:
        raw_response = None
        for attempt in range(self.config.max_retries):
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
                raw_response = response.choices[0].message.content.strip()
                result = self._extract_json_object(raw_response)
                for key in SCORE_KEYS:
                    value = result.get(key, 3)
                    try:
                        value = int(value)
                    except Exception:
                        value = 3
                    result[key] = max(1, min(5, value))
                result["overall_judge"] = compute_overall(result)
                return result
            except RateLimitError:
                wait_sec = self.config.retry_base_wait * (attempt + 1)
                print(f"[RateLimit] retry {attempt + 1}/{self.config.max_retries}, waiting {wait_sec}s")
                time.sleep(wait_sec)
            except json.JSONDecodeError as exc:
                print(f"[ERROR] JSON parse failed: {exc} | raw={raw_response}")
                return None
            except Exception as exc:
                print(f"[ERROR] API call failed: {exc}")
                return None
        return None


class IterMADPipeline:
    def __init__(self, client: JudgeClient, n_iterations: int):
        self.client = client
        self.n_iterations = n_iterations

    def _strict_initial(self, sample: dict) -> dict | None:
        return self.client.call_json(
            STRICT_JUDGE_SYSTEM,
            build_user_prompt(sample) + "\n\nOutput JSON only.",
            temperature=0.1,
        )

    def _lenient_initial(self, sample: dict) -> dict | None:
        return self.client.call_json(
            LENIENT_JUDGE_SYSTEM,
            build_user_prompt(sample) + "\n\nOutput JSON only.",
            temperature=0.1,
        )

    def _strict_adjust(self, sample: dict, other_result: dict) -> dict | None:
        prompt = build_adjust_prompt(
            sample,
            other_result,
            reviewer_label="lenient_judge",
            instruction="Revise your own scores if needed. Output JSON only.",
        )
        return self.client.call_json(STRICT_JUDGE_ADJUST_SYSTEM, prompt, temperature=0.0)

    def _lenient_adjust(self, sample: dict, other_result: dict) -> dict | None:
        prompt = build_adjust_prompt(
            sample,
            other_result,
            reviewer_label="strict_judge",
            instruction="Revise your own scores if needed. Output JSON only.",
        )
        return self.client.call_json(LENIENT_JUDGE_ADJUST_SYSTEM, prompt, temperature=0.0)

    def _future_result_or_none(self, future) -> dict | None:
        try:
            return future.result()
        except Exception as exc:
            print(f"[ERROR] worker failed: {exc}")
            return None

    def run(self, sample: dict) -> dict | None:
        rounds: list[dict] = [{} for _ in range(self.n_iterations)]

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_meta = {
                executor.submit(self._strict_initial, sample): ("strict", 0),
                executor.submit(self._lenient_initial, sample): ("lenient", 0),
            }

            while future_meta:
                done, _ = wait(future_meta.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    side, round_num = future_meta.pop(future)
                    result = self._future_result_or_none(future)
                    if not result:
                        for pending in future_meta:
                            pending.cancel()
                        return None

                    rounds[round_num][side] = result
                    next_round = round_num + 1
                    if next_round >= self.n_iterations:
                        continue

                    if side == "strict":
                        next_future = executor.submit(self._lenient_adjust, sample, result)
                        future_meta[next_future] = ("lenient", next_round)
                    else:
                        next_future = executor.submit(self._strict_adjust, sample, result)
                        future_meta[next_future] = ("strict", next_round)

        last_round = rounds[-1]
        if "strict" not in last_round or "lenient" not in last_round:
            return None

        final = {
            key: round((last_round["strict"].get(key, 3) + last_round["lenient"].get(key, 3)) / 2, 1)
            for key in SCORE_KEYS
        }
        final["overall_judge"] = round(
            (last_round["strict"]["overall_judge"] + last_round["lenient"]["overall_judge"]) / 2,
            1,
        )

        output = {f"round_{idx}": round_result for idx, round_result in enumerate(rounds)}
        output["final"] = final
        return output


def process_item(input_path: Path, output_path: Path, judge_model: str, pipeline: IterMADPipeline) -> str:
    if output_path.exists():
        return f"[SKIP] {output_path}"

    try:
        sample = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"[ERROR] read failed {input_path}: {exc}"

    result = pipeline.run(sample)
    if not result:
        return f"[ERROR] MAD-A-iter failed {input_path}"

    output = {
        "article_id": sample.get("article_id"),
        "system_name": sample.get("system_name"),
        "gold": sample.get("gold", {}),
        "judge_model": judge_model,
        "judge": result,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"[DONE] {output_path}"


def discover_pending_files(output_root: Path) -> list[tuple[Path, Path]]:
    pending = []
    if not INPUT_BASE.exists():
        return pending

    for system_dir in sorted(INPUT_BASE.iterdir()):
        if not system_dir.is_dir():
            continue
        for input_path in sorted(system_dir.glob("*.json")):
            output_path = output_root / system_dir.name / input_path.name
            if not output_path.exists():
                pending.append((input_path, output_path))
    return pending


def main():
    parser = argparse.ArgumentParser(description="MAD-A iterative SummEval evaluation pipeline")
    parser.add_argument("--judge-model", default="gpt", choices=["gpt", "gemma", "qwen"])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    config = IterConfig(judge_model=args.judge_model, n_iterations=args.iterations)
    config.validate()

    client = JudgeClient(config)
    pipeline = IterMADPipeline(client, n_iterations=config.n_iterations)
    output_root = OUTPUT_BASE / args.judge_model / f"iter{args.iterations}"
    output_root.mkdir(parents=True, exist_ok=True)

    pending = discover_pending_files(output_root)
    print(f"Judge model: {client.model_name}")
    print(f"Iterations: {args.iterations}")
    print(f"Pending items: {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(process_item, input_path, output_path, args.judge_model, pipeline)
            for input_path, output_path in pending
        ]
        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()
