import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

# 전역 기본 경로 (수정 가능)
INPUT_PATH = "results/results_llama3_8b"
OUTPUT_PATH = "llama3_8b.json"

PERFORMANCE_CRITERIA = ["evaluator1", "evaluator2", "average"]
AREAS = ["content", "organization", "expression"]

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


# -----------------------------
# Spearman 계산 함수
# -----------------------------
def rankdata(values: List[float]) -> List[float]:
    """평균 순위 방식"""
    sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0] * len(values)

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


def spearman_correlation(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 2:
        return None

    rx = rankdata(x)
    ry = rankdata(y)

    mean_rx = sum(rx) / len(rx)
    mean_ry = sum(ry) / len(ry)

    num = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mean_rx) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - mean_ry) ** 2 for b in ry))

    if den_x == 0 or den_y == 0:
        return None

    return num / (den_x * den_y)


# -----------------------------
# RMSE + Spearman 계산
# -----------------------------
def compute_metrics(records: List[Dict[str, Any]]):
    rmse_accum = {
        crit: {area: [] for area in AREAS} for crit in PERFORMANCE_CRITERIA
    }

    spearman_pairs = {
        crit: {area: {"gold": [], "pred": []} for area in AREAS}
        for crit in PERFORMANCE_CRITERIA
    }

    for rec in records:
        gold_block = rec.get("gold", {})
        pred_block = rec.get("prediction", {})

        for crit in PERFORMANCE_CRITERIA:
            gold_scores = gold_block.get(crit, {})

            for area in AREAS:
                gold_val = safe_float(gold_scores.get(area))

                pred_area = pred_block.get(area, {})
                if isinstance(pred_area, dict):
                    pred_val = safe_float(pred_area.get("score"))
                else:
                    pred_val = safe_float(pred_area)

                if gold_val is None or pred_val is None:
                    continue

                # RMSE
                diff = pred_val - gold_val
                rmse_accum[crit][area].append(diff * diff)

                # Spearman
                spearman_pairs[crit][area]["gold"].append(gold_val)
                spearman_pairs[crit][area]["pred"].append(pred_val)

    # 결과 정리
    rmse_result = {}
    spearman_result = {}

    for crit in PERFORMANCE_CRITERIA:
        rmse_result[crit] = {}
        spearman_result[crit] = {}

        for area in AREAS:
            # RMSE
            seq = rmse_accum[crit][area]
            if not seq:
                rmse_result[crit][area] = None
            else:
                rmse_result[crit][area] = math.sqrt(sum(seq) / len(seq))

            # Spearman
            gold_list = spearman_pairs[crit][area]["gold"]
            pred_list = spearman_pairs[crit][area]["pred"]

            spearman_result[crit][area] = spearman_correlation(gold_list, pred_list)

    return rmse_result, spearman_result


def collect_records_from_folder(input_dir: Path) -> List[Dict[str, Any]]:
    all_records: List[Dict[str, Any]] = []

    for p in sorted(input_dir.glob("*.json")):
        try:
            data = load_json(p)
        except Exception as e:
            print(f"[WARNING] JSON 읽기 실패 {p}: {e}")
            continue

        if isinstance(data, list):
            all_records.extend(data)
        elif isinstance(data, dict):
            all_records.append(data)
        else:
            print(f"[WARNING] 예상치 못한 JSON 구조 {p}")

    return all_records


def main():
    parser = argparse.ArgumentParser(description="LLM 채점 RMSE + Spearman 산출")
    parser.add_argument("--input-dir", default=INPUT_PATH)
    parser.add_argument("--output-file", default=OUTPUT_PATH)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    records = collect_records_from_folder(input_dir)

    rmse_metrics, spearman_metrics = compute_metrics(records)

    output_data = {
        "total_documents": len(records),
        "rmse": rmse_metrics,
        "spearman": spearman_metrics
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("========== RESULT ==========")
    for crit in PERFORMANCE_CRITERIA:
        print(f"\n== {crit} ==")
        for area in AREAS:
            print(
                f"{area} | RMSE: {rmse_metrics[crit][area]:.4f} "
                f"| Spearman: {spearman_metrics[crit][area]}"
            )


if __name__ == "__main__":
    main()