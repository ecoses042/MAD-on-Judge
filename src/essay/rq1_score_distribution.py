"""
Essay automatic scoring distribution metrics.

Final essay direction in outline.md is direct essay scoring:
  prompt_text + essay_text -> content/organization/expression scores.

This script therefore aggregates score distributions from direct scoring outputs
under inference_results/{model}/ instead of meta-judge outputs under judge_results/.
Use essay_scoring_performance.py for gold-label performance metrics such as RMSE and Spearman.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


AREAS = ["content", "organization", "expression"]
SCORE_KEYS = [*AREAS, "overall"]
DEFAULT_INPUT_ROOT = "inference_results"
DEFAULT_OUTPUT = "stats/essay_direct_distribution_metrics.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def iter_records(file_path: Path) -> list[dict[str, Any]]:
    data = load_json(file_path)
    records = data if isinstance(data, list) else [data]
    return [record for record in records if isinstance(record, dict)]


def extract_prediction_scores(record: dict[str, Any]) -> dict[str, float | None]:
    prediction = record.get("prediction", {})
    scores: dict[str, float | None] = {area: None for area in AREAS}
    if not isinstance(prediction, dict):
        scores["overall"] = None
        return scores

    for area in AREAS:
        area_block = prediction.get(area)
        if isinstance(area_block, dict):
            scores[area] = safe_float(area_block.get("score"))
        else:
            scores[area] = safe_float(area_block)

    valid = [scores[area] for area in AREAS if scores[area] is not None]
    scores["overall"] = round(sum(valid) / len(valid), 4) if valid else None
    return scores


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]

    pos = (len(sorted_values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[int(pos)]
    weight = pos - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize_distribution(scores: list[float]) -> dict[str, Any] | None:
    if not scores:
        return None

    n = len(scores)
    mean = sum(scores) / n
    variance = sum((score - mean) ** 2 for score in scores) / n
    std = math.sqrt(variance)
    sorted_scores = sorted(scores)
    q1 = percentile(sorted_scores, 0.25)
    q3 = percentile(sorted_scores, 0.75)

    rounded_counts = Counter(round(score) for score in scores)
    probabilities = [count / n for count in rounded_counts.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    normalized_entropy = entropy / math.log2(5)

    return {
        "count": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "range": round(max(scores) - min(scores), 4),
        "iqr": round(q3 - q1, 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "normalized_entropy": round(normalized_entropy, 4),
        "mode_freq": round(max(rounded_counts.values()) / n, 4),
        "score_counts_rounded": {str(key): rounded_counts[key] for key in sorted(rounded_counts)},
    }


def collect_model_scores(model_dir: Path) -> dict[str, list[float]]:
    accum: dict[str, list[float]] = {key: [] for key in SCORE_KEYS}

    for file_path in sorted(model_dir.glob("*.json")):
        try:
            records = iter_records(file_path)
        except Exception as e:
            print(f"[WARNING] 읽기 실패 {file_path}: {e}")
            continue

        for record in records:
            scores = extract_prediction_scores(record)
            for key in SCORE_KEYS:
                value = scores.get(key)
                if value is not None:
                    accum[key].append(value)

    return accum


def discover_models(input_root: Path, selected_models: list[str] | None) -> list[str]:
    if selected_models:
        return selected_models
    if not input_root.exists():
        return []
    return sorted(path.name for path in input_root.iterdir() if path.is_dir())


def aggregate(input_root: Path, models: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model in models:
        model_dir = input_root / model
        if not model_dir.exists():
            print(f"[WARNING] 모델 디렉토리 없음: {model_dir}")
            continue

        print(f"[INFO] 직접채점 분포 집계: {model_dir}")
        accum = collect_model_scores(model_dir)
        model_result = {
            key: summarize_distribution(values)
            for key, values in accum.items()
        }
        result[model] = model_result
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Essay direct automatic scoring distribution metrics")
    parser.add_argument("--input-root", type=Path, default=Path(DEFAULT_INPUT_ROOT))
    parser.add_argument("--models", nargs="+", default=None, help="집계할 모델 디렉토리 이름. 생략하면 전체")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = discover_models(args.input_root, args.models)
    if not models:
        raise FileNotFoundError(f"No model directories found under {args.input_root}")

    metrics = aggregate(args.input_root, models)
    output = {
        "settings": {
            "input_root": str(args.input_root),
            "models": models,
            "score_keys": SCORE_KEYS,
        },
        "metrics": metrics,
    }
    write_json(output, args.output)

    for model, model_metrics in metrics.items():
        print(f"\n== {model} ==")
        for key in SCORE_KEYS:
            row = model_metrics.get(key)
            if not row:
                print(f"{key}: n=0")
                continue
            print(
                f"{key}: n={row['count']}, mean={row['mean']}, std={row['std']}, "
                f"range={row['range']}, iqr={row['iqr']}, mode_freq={row['mode_freq']}"
            )
    print(f"\n[INFO] 저장 완료: {args.output}")


if __name__ == "__main__":
    main()
