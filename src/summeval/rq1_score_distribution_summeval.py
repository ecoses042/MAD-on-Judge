import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # adds src/ to path
from env_utils import load_project_env

load_project_env(__file__)

import argparse
import json
import math
from collections import Counter

try:
    from scipy.stats import wasserstein_distance

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

SCORE_KEYS = ["coherence", "consistency", "fluency", "relevance"]
RESULT_BASE = Path("summeval_judge_results")
SINGLE_JUDGE_BASE = RESULT_BASE / "single_judge"
MAD1_BASE = RESULT_BASE / "mad1"
MAD2_ITER_BASE = RESULT_BASE / "mad2_iter"


def register_experiments() -> dict[str, tuple[Path, bool]]:
    registry: dict[str, tuple[Path, bool]] = {}

    if SINGLE_JUDGE_BASE.exists():
        for judge_dir in sorted(SINGLE_JUDGE_BASE.iterdir()):
            if judge_dir.is_dir():
                registry[f"single_judge/{judge_dir.name}"] = (judge_dir, False)

    if MAD1_BASE.exists():
        for judge_dir in sorted(MAD1_BASE.iterdir()):
            if judge_dir.is_dir():
                registry[f"mad1/{judge_dir.name}"] = (judge_dir, False)

    if MAD2_ITER_BASE.exists():
        for judge_dir in sorted(MAD2_ITER_BASE.iterdir()):
            if not judge_dir.is_dir():
                continue
            for iter_dir in sorted(judge_dir.iterdir()):
                if iter_dir.is_dir():
                    registry[f"mad2_iter/{judge_dir.name}/{iter_dir.name}"] = (iter_dir, True)

    return registry


def collect_scores_from_files(files: list[Path], uses_final: bool) -> dict[str, list[float]]:
    accum = {key: [] for key in SCORE_KEYS}

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARNING] failed to read {path}: {exc}")
            continue

        judge_block = data.get("judge", {})
        score_block = judge_block.get("final", {}) if uses_final else judge_block
        if uses_final and not isinstance(score_block, dict):
            continue
        if not uses_final and "final" in judge_block and all(k in judge_block["final"] for k in SCORE_KEYS):
            score_block = judge_block["final"]

        for key in SCORE_KEYS:
            value = score_block.get(key)
            if value is None:
                continue
            try:
                accum[key].append(float(value))
            except (TypeError, ValueError):
                continue

    return accum


def compute_metrics(scores: list[float]) -> dict | None:
    if not scores:
        return None

    n = len(scores)
    mean = sum(scores) / n
    variance = sum((x - mean) ** 2 for x in scores) / n
    std = math.sqrt(variance)
    sorted_scores = sorted(scores)

    q1 = sorted_scores[int((n - 1) * 0.25)]
    q3 = sorted_scores[int((n - 1) * 0.75)]
    score_range = max(scores) - min(scores)
    iqr = q3 - q1
    cv = std / mean if mean else 0.0

    counts = Counter(round(score) for score in scores)
    probs = [count / n for count in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    normalized_entropy = entropy / math.log2(5) if math.log2(5) > 0 else 0.0
    mode_freq = max(counts.values()) / n if counts else 0.0

    result = {
        "count": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "range": round(score_range, 4),
        "iqr": round(iqr, 4),
        "cv": round(cv, 4),
        "normalized_entropy": round(normalized_entropy, 4),
        "mode_freq": round(mode_freq, 4),
    }

    if HAS_SCIPY:
        actual = [counts.get(i, 0) / n for i in range(1, 6)]
        uniform = [1 / 5] * 5
        wd = wasserstein_distance(range(1, 6), range(1, 6), actual, uniform)
        result["wasserstein_from_uniform"] = round(wd, 4)

    return result


def aggregate_experiment(exp_key: str, exp_dir: Path, uses_final: bool) -> dict | None:
    if not exp_dir.exists():
        return None

    result = {}
    for system_dir in sorted(exp_dir.iterdir()):
        if not system_dir.is_dir():
            continue
        files = sorted(system_dir.glob("*.json"))
        if not files:
            continue
        scores = collect_scores_from_files(files, uses_final=uses_final)
        result[system_dir.name] = {key: compute_metrics(values) for key, values in scores.items()}

    return result or None


def main():
    parser = argparse.ArgumentParser(description="Aggregate SummEval experiment metrics")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exp", help="single_judge/gpt, mad1/gpt, mad2_iter/gpt/iter3, etc.")
    group.add_argument("--all", action="store_true", help="Aggregate all discovered experiments")
    parser.add_argument("--output", default="stats/summeval_metrics_all.json")
    args = parser.parse_args()

    registry = register_experiments()
    if not registry:
        print("[WARNING] No experiments found under summeval_judge_results/")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    if args.all:
        final_output = {}
        for exp_key, (exp_dir, uses_final) in registry.items():
            agg = aggregate_experiment(exp_key, exp_dir, uses_final=uses_final)
            if agg:
                final_output[exp_key] = agg
                print(f"[DONE] {exp_key}")
        output_path.write_text(json.dumps(final_output, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        if args.exp not in registry:
            raise ValueError(f"Unknown experiment: {args.exp}")
        exp_dir, uses_final = registry[args.exp]
        agg = aggregate_experiment(args.exp, exp_dir, uses_final=uses_final)
        if not agg:
            print(f"[ERROR] failed to aggregate {args.exp}")
            return
        existing[args.exp] = agg
        output_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        final_output = existing
        print(f"[DONE] {args.exp}")

    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
