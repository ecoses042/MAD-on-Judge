import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

# 기본 경로 설정
INPUT_DIR = [
    # "judge_results/single_judge/gemma",
    # "judge_results/single_judge/qwen",
    # "judge_results/single_judge/llama",
    # "judge_results/single_judge/gpt",
    # "judge_results/mad_c/gemma",
    # "judge_results/mad_c/qwen",
    # "judge_results/mad_c/llama",
    # "judge_results/mad_c/gpt",
    "judge_results/mad_c_seq/gemma",
    "judge_results/mad_c_seq/qwen",
    "judge_results/mad_c_seq/llama",
    "judge_results/mad_c_seq/gpt",
]

AREAS = ["content", "organization", "expression"]
SCORE_KEYS = ["domain_match", "score_rationale_consistency", "specificity", "groundedness"]


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def collect_judge_records_from_folder(input_dir: Path) -> Dict[str, List[float]]:
    judge_accum = {key: [] for key in SCORE_KEYS}

    for p in sorted(input_dir.glob("*.json")):
        try:
            data = load_json(p)
        except Exception as e:
            print(f"[WARNING] JSON 읽기 실패 {p}: {e}")
            continue

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = [data]
        else:
            print(f"[WARNING] 예상치 못한 JSON 구조 {p}")
            continue

        for rec in records:
            judge_block = rec.get("judge", {})
            for area in AREAS:
                judge_area = judge_block.get(area, {})
                if not isinstance(judge_area, dict):
                    continue
                for key in SCORE_KEYS:
                    val = safe_float(judge_area.get(key))
                    if val is not None:
                        judge_accum[key].append(val)

    return judge_accum


def compute_mean(seq: List[float]) -> Optional[float]:
    if not seq:
        return None
    return sum(seq) / len(seq)


def compute_std(seq: List[float]) -> Optional[float]:
    if not seq:
        return None
    mean = sum(seq) / len(seq)
    var = sum((x - mean) ** 2 for x in seq) / len(seq)
    return math.sqrt(var)


def compute_score_counts(seq: List[float]) -> Dict[str, int]:
    counts = {str(i): 0 for i in range(1, 6)}
    for x in seq:
        score_int = int(round(x))
        if 1 <= score_int <= 5:
            counts[str(score_int)] += 1
    return counts


def compute_judge_stats(judge_accum: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
    stats = {}

    for key in SCORE_KEYS:
        seq = judge_accum[key]

        stats[key] = {
            "count": len(seq),
            "mean": compute_mean(seq),
            "std": compute_std(seq),
            "score_counts": compute_score_counts(seq),
            "histogram_0.5": compute_histogram(seq, 0.5),
            "histogram_0.2": compute_histogram(seq, 0.2),
        }

    return stats


def compute_histogram(seq: List[float], bin_size: float = 0.5) -> Dict[str, int]:
    if not seq:
        return {}

    min_val = min(seq)
    max_val = max(seq)

    start = math.floor(min_val / bin_size) * bin_size
    end = math.ceil(max_val / bin_size) * bin_size

    bins = {}
    current = start

    while current < end + 1e-9:
        bin_label = f"{current:.1f}-{current + bin_size:.1f}"
        bins[bin_label] = 0
        current = round(current + bin_size, 10)

    for x in seq:
        idx = math.floor((x - start) / bin_size)
        bin_start = start + idx * bin_size
        bin_label = f"{bin_start:.1f}-{bin_start + bin_size:.1f}"
        if bin_label in bins:
            bins[bin_label] += 1
        else:
            last_key = list(bins.keys())[-1]
            bins[last_key] += 1

    return bins


def main():
    parser = argparse.ArgumentParser(description="Judge score 통계 계산 (모델별 분리)")
    parser.add_argument("--input-dirs", nargs="+", default=INPUT_DIR, help="입력 디렉토리 경로들")
    parser.add_argument("--output-prefix", default="stats/judge_mad_c_seq", help="출력 파일 접두사")
    args = parser.parse_args()

    input_dirs = args.input_dirs if isinstance(args.input_dirs, list) else [args.input_dirs]

    all_results = {}

    for input_dir_name in input_dirs:
        input_dir = Path(input_dir_name)
        if not input_dir.exists():
            print(f"[WARNING] 입력 디렉토리가 존재하지 않음: {input_dir}")
            continue

        print(f"[INFO] 처리 중: {input_dir}")

        model_name = input_dir.name

        judge_accum = collect_judge_records_from_folder(input_dir)
        judge_stats = compute_judge_stats(judge_accum)

        output_data = {
            "model": model_name,
            "judge_stats": judge_stats,
        }

        all_results[f"mad_c_seq_results_{model_name}"] = output_data

        output_file = f"{args.output_prefix}_{model_name}.json"
        Path(output_file).parent.mkdir(exist_ok=True, parents=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"[INFO] 저장됨: {output_file}")
        print(f"-------- {model_name} --------")
        for key in SCORE_KEYS:
            key_stats = judge_stats[key]
            mean = key_stats["mean"]
            std = key_stats["std"]
            count = key_stats["count"]
            score_counts = key_stats["score_counts"]

            if mean is not None and std is not None:
                print(
                    f"  {key} | Count: {count} | Mean: {mean:.4f} | Std: {std:.4f} "
                    f"| Counts: {score_counts}"
                )
            else:
                print(f"  {key} | Count: {count} | Mean: N/A | Std: N/A | Counts: {score_counts}")

    combined_output = f"{args.output_prefix}_all.json"
    with open(combined_output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] 전체 결과 저장됨: {combined_output}")


if __name__ == "__main__":
    main()
