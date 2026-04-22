import json
from pathlib import Path
from typing import Any, Dict, List, Optional

INPUT_DIRS = [
    "judge_results/mad_a_base/gemma",
    "judge_results/mad_a_base/qwen",
    "judge_results/mad_a_base/llama",
    "judge_results/mad_a_base/gpt",
]
OUTPUT_FILE = "stats/mad_a_base_stats_all.json"

AREAS     = ["content", "organization", "expression"]
SCORE_KEYS = ["domain_match", "score_rationale_consistency", "specificity", "groundedness"]


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean(seq: List[float]) -> Optional[float]:
    if not seq:
        return None
    return round(sum(seq) / len(seq), 4)


def collect_from_folder(input_dir: Path) -> Dict[str, List[float]]:
    """{ score_key: [float, ...] } — final stage, 3개 영역 풀링"""
    accum = {key: [] for key in SCORE_KEYS}

    for p in sorted(input_dir.glob("*.json")):
        try:
            data = load_json(p)
        except Exception as e:
            print(f"[WARNING] 읽기 실패 {p}: {e}")
            continue

        records = data if isinstance(data, list) else [data]
        for rec in records:
            for area in AREAS:
                final_block = rec.get("judge", {}).get(area, {}).get("final", {})
                if not isinstance(final_block, dict):
                    continue
                for key in SCORE_KEYS:
                    val = final_block.get(key)
                    if val is not None:
                        try:
                            accum[key].append(float(val))
                        except (TypeError, ValueError):
                            pass

    return accum


def main():
    all_results: Dict[str, Any] = {}
    Path(OUTPUT_FILE).parent.mkdir(exist_ok=True, parents=True)

    for input_dir_name in INPUT_DIRS:
        input_dir = Path(input_dir_name)
        if not input_dir.exists():
            print(f"[WARNING] 디렉토리 없음, 건너뜀: {input_dir}")
            continue

        print(f"[INFO] 처리 중: {input_dir}")
        accum = collect_from_folder(input_dir)

        model_stats = {key: mean(accum[key]) for key in SCORE_KEYS}
        all_results[f"mad_a_base_results_{input_dir.name}"] = model_stats

        print(f"-------- {input_dir.name} --------")
        for key, val in model_stats.items():
            print(f"  {key:<35} | {val:.4f}" if val is not None else f"  {key:<35} | N/A")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] 저장 완료: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
