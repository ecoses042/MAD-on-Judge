"""
Essay automatic scoring metric aggregation.

This script evaluates direct essay score predictions against NIKL human labels.
It uses evaluator average as the default gold label, matching outline.md.

Supported record shapes:
- inference result records with gold.average.{content,organization,expression}
- prepared essay records with label_5scale_average.{con,org,exp}
- inference result records without gold, joined by --essay-data-dir on essay_id
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


AREAS = ["content", "organization", "expression"]
AREA_TO_LABEL = {
    "content": "con",
    "organization": "org",
    "expression": "exp",
}
PERFORMANCE_CRITERIA = ["average", "evaluator1", "evaluator2"]
DEFAULT_INPUT_DIR = "inference_results/gemma"
DEFAULT_ESSAY_DATA_DIR = "data/selected_prompt_jsons_100"
DEFAULT_OUTPUT_FILE = "stats/essay_scoring_metrics.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def iter_json_records(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    records = data if isinstance(data, list) else [data]
    return [record for record in records if isinstance(record, dict)]


def collect_records_from_folder(input_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for json_file in sorted(input_dir.glob("*.json")):
        try:
            records.extend(iter_json_records(json_file))
        except Exception as e:
            print(f"[WARNING] JSON 읽기 실패 {json_file}: {e}")
    return records


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


def normalize_area_label(label_block: dict[str, Any] | None) -> dict[str, float | None]:
    if not isinstance(label_block, dict):
        return {area: None for area in AREAS}
    return {
        area: safe_float(label_block.get(area, label_block.get(AREA_TO_LABEL[area])))
        for area in AREAS
    }


def build_essay_gold_index(essay_data_dir: Path) -> dict[str, dict[str, dict[str, float | None]]]:
    index: dict[str, dict[str, dict[str, float | None]]] = {}
    if not essay_data_dir.exists():
        return index

    for json_file in sorted(essay_data_dir.glob("*.json")):
        try:
            records = iter_json_records(json_file)
        except Exception as e:
            print(f"[WARNING] essay index 읽기 실패 {json_file}: {e}")
            continue

        for record in records:
            essay_id = record.get("essay_id")
            if not essay_id:
                continue
            index[str(essay_id)] = {
                "average": normalize_area_label(record.get("label_5scale_average")),
                "evaluator1": normalize_area_label(record.get("label_5scale_evaluator1")),
                "evaluator2": normalize_area_label(record.get("label_5scale_evaluator2")),
            }
    return index


def extract_gold(
    record: dict[str, Any],
    criterion: str,
    essay_gold_index: dict[str, dict[str, dict[str, float | None]]],
) -> dict[str, float | None]:
    gold = record.get("gold")
    if isinstance(gold, dict) and isinstance(gold.get(criterion), dict):
        return normalize_area_label(gold[criterion])

    label_key = "label_5scale_average" if criterion == "average" else f"label_5scale_{criterion}"
    if isinstance(record.get(label_key), dict):
        return normalize_area_label(record[label_key])

    essay_id = record.get("essay_id")
    if essay_id and str(essay_id) in essay_gold_index:
        return essay_gold_index[str(essay_id)].get(criterion, normalize_area_label(None))

    return normalize_area_label(None)


def extract_prediction(record: dict[str, Any]) -> dict[str, float | None]:
    prediction = record.get("prediction", {})
    if not isinstance(prediction, dict):
        return {area: None for area in AREAS}

    scores: dict[str, float | None] = {}
    for area in AREAS:
        value = prediction.get(area)
        if isinstance(value, dict):
            scores[area] = safe_float(value.get("score"))
        else:
            scores[area] = safe_float(value)
    return scores


def rankdata(values: list[float]) -> list[float]:
    sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[sorted_idx[j]] == values[sorted_idx[j + 1]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks


def correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    denominator_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    if denominator_x == 0 or denominator_y == 0:
        return None
    return numerator / (denominator_x * denominator_y)


def spearman_correlation(x: list[float], y: list[float]) -> float | None:
    return correlation(rankdata(x), rankdata(y))


def summarize_pairs(pairs: list[tuple[float, float]]) -> dict[str, float | int | None]:
    if not pairs:
        return {
            "n": 0,
            "rmse": None,
            "mae": None,
            "pearson": None,
            "spearman": None,
            "exact_match_rate": None,
            "within_0_5_rate": None,
            "within_1_0_rate": None,
            "bias_mean_pred_minus_gold": None,
        }

    gold = [g for g, _ in pairs]
    pred = [p for _, p in pairs]
    errors = [p - g for g, p in pairs]
    abs_errors = [abs(err) for err in errors]
    n = len(pairs)
    return {
        "n": n,
        "rmse": round(math.sqrt(sum(err * err for err in errors) / n), 4),
        "mae": round(sum(abs_errors) / n, 4),
        "pearson": round(correlation(gold, pred), 4) if correlation(gold, pred) is not None else None,
        "spearman": round(spearman_correlation(gold, pred), 4)
        if spearman_correlation(gold, pred) is not None
        else None,
        "exact_match_rate": round(sum(1 for err in abs_errors if err == 0) / n, 4),
        "within_0_5_rate": round(sum(1 for err in abs_errors if err <= 0.5) / n, 4),
        "within_1_0_rate": round(sum(1 for err in abs_errors if err <= 1.0) / n, 4),
        "bias_mean_pred_minus_gold": round(sum(errors) / n, 4),
    }


def mean_valid(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def compute_metrics(
    records: list[dict[str, Any]],
    essay_gold_index: dict[str, dict[str, dict[str, float | None]]],
    criteria: list[str],
) -> dict[str, Any]:
    pairs: dict[str, dict[str, list[tuple[float, float]]]] = {
        criterion: {area: [] for area in [*AREAS, "overall"]}
        for criterion in criteria
    }

    for record in records:
        pred_scores = extract_prediction(record)
        pred_overall = mean_valid([pred_scores[area] for area in AREAS])

        for criterion in criteria:
            gold_scores = extract_gold(record, criterion, essay_gold_index)
            gold_overall = mean_valid([gold_scores[area] for area in AREAS])

            for area in AREAS:
                gold_val = gold_scores[area]
                pred_val = pred_scores[area]
                if gold_val is not None and pred_val is not None:
                    pairs[criterion][area].append((gold_val, pred_val))

            if gold_overall is not None and pred_overall is not None:
                pairs[criterion]["overall"].append((gold_overall, pred_overall))

    return {
        criterion: {
            area: summarize_pairs(area_pairs)
            for area, area_pairs in area_map.items()
        }
        for criterion, area_map in pairs.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Essay automatic scoring metrics")
    parser.add_argument("--input-dir", type=Path, default=Path(DEFAULT_INPUT_DIR))
    parser.add_argument("--essay-data-dir", type=Path, default=Path(DEFAULT_ESSAY_DATA_DIR))
    parser.add_argument("--output-file", type=Path, default=Path(DEFAULT_OUTPUT_FILE))
    parser.add_argument(
        "--gold-source",
        choices=[*PERFORMANCE_CRITERIA, "all"],
        default="average",
        help="Human gold label source. outline.md default is average.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(args.input_dir)

    criteria = PERFORMANCE_CRITERIA if args.gold_source == "all" else [args.gold_source]
    records = collect_records_from_folder(args.input_dir)
    essay_gold_index = build_essay_gold_index(args.essay_data_dir)
    metrics = compute_metrics(records, essay_gold_index, criteria)

    output = {
        "settings": {
            "input_dir": str(args.input_dir),
            "essay_data_dir": str(args.essay_data_dir),
            "gold_source": args.gold_source,
            "records": len(records),
            "essay_gold_index_size": len(essay_gold_index),
        },
        "metrics": metrics,
    }
    write_json(output, args.output_file)

    print(f"[INFO] records={len(records)}, essay_gold_index={len(essay_gold_index)}")
    for criterion, criterion_metrics in metrics.items():
        print(f"\n== {criterion} ==")
        for area in [*AREAS, "overall"]:
            row = criterion_metrics[area]
            print(
                f"{area}: n={row['n']}, rmse={row['rmse']}, mae={row['mae']}, "
                f"spearman={row['spearman']}, within_1.0={row['within_1_0_rate']}"
            )
    print(f"\n[INFO] 저장 완료: {args.output_file}")


if __name__ == "__main__":
    main()
