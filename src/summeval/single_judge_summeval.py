import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # adds src/ to path
from env_utils import load_project_env

load_project_env(__file__)

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI, RateLimitError

from summeval.summeval_prompt_utils import SCORE_KEYS, SINGLE_JUDGE_SYSTEM, build_user_prompt, compute_overall

API_KEY = os.environ.get("OPENAI_API_KEY", "")
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODELS = {
    "gemma": "google/gemma-3-4b",
    "qwen": "qwen/qwen3.5-9b",
}
INPUT_BASE = Path("summeval_judge_input")
OUTPUT_BASE = Path("summeval_judge_results/single_judge")


def make_client(judge_model: str):
    if judge_model in LM_STUDIO_MODELS:
        client = OpenAI(api_key="lm-studio", base_url=LM_STUDIO_BASE_URL)
        model_name = LM_STUDIO_MODELS[judge_model]
        return client, model_name
    client = OpenAI(api_key=API_KEY)
    return client, "gpt-4o-mini"


def extract_json_object(raw_text: str) -> dict:
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


def call_judge(client: OpenAI, model_name: str, sample: dict, max_retries: int = 3) -> dict | None:
    use_json_format = model_name.startswith("gpt-")
    raw_response = None

    for attempt in range(max_retries):
        try:
            create_kwargs = dict(
                model=model_name,
                messages=[
                    {"role": "system", "content": SINGLE_JUDGE_SYSTEM},
                    {"role": "user", "content": build_user_prompt(sample)},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            if use_json_format:
                create_kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**create_kwargs)
            raw_response = response.choices[0].message.content.strip()
            result = extract_json_object(raw_response)

            for key in SCORE_KEYS:
                value = result.get(key)
                try:
                    value = int(value)
                except Exception:
                    return None
                result[key] = max(1, min(5, value))

            result["overall_judge"] = compute_overall(result)
            return result
        except RateLimitError:
            wait_sec = 5 * (attempt + 1)
            print(f"[RateLimit] retry {attempt + 1}/{max_retries}, waiting {wait_sec}s")
            time.sleep(wait_sec)
        except json.JSONDecodeError as exc:
            print(f"[ERROR] JSON parse failed: {exc} | raw={raw_response}")
            return None
        except Exception as exc:
            print(f"[ERROR] API call failed: {exc}")
            return None

    return None


def process_item(
    input_path: Path,
    output_path: Path,
    judge_model: str,
    client: OpenAI,
    model_name: str,
) -> str:
    if output_path.exists():
        return f"[SKIP] {output_path}"

    try:
        sample = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"[ERROR] read failed {input_path}: {exc}"

    judge_result = call_judge(client, model_name, sample)
    if not judge_result:
        return f"[ERROR] judge failed {input_path}"

    output = {
        "article_id": sample.get("article_id"),
        "system_name": sample.get("system_name"),
        "gold": sample.get("gold", {}),
        "judge_model": judge_model,
        "judge": judge_result,
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
    parser = argparse.ArgumentParser(description="Single-judge SummEval evaluation pipeline")
    parser.add_argument("--judge-model", default="gpt", choices=["gpt", "gemma", "qwen"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.judge_model == "gpt" and not API_KEY:
        raise ValueError("OPENAI_API_KEY is not set.")

    client, model_name = make_client(args.judge_model)
    output_root = OUTPUT_BASE / args.judge_model
    output_root.mkdir(parents=True, exist_ok=True)

    pending = discover_pending_files(output_root)
    print(f"Judge model: {model_name}")
    print(f"Pending items: {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(process_item, input_path, output_path, args.judge_model, client, model_name)
            for input_path, output_path in pending
        ]
        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()
