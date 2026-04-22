"""
get_metrics.py — outline.md 기준 Judge 실험 메트릭 집계

지원 실험:
  single_judge/{judge_model}    : judge_results/single_judge/{judge_model}/{essay_model}/
  mad_c                         : judge_results/mad_c/{essay_model}/
  mad_a_iter/{judge_model}/iter3: judge_results/mad_a_iter/{judge_model}/iter3/{essay_model}/
  mad_a_iter/{judge_model}/iter5: judge_results/mad_a_iter/{judge_model}/iter5/{essay_model}/

CLI:
  python src/get_metrics.py --exp single_judge/gpt
  python src/get_metrics.py --exp mad_c
  python src/get_metrics.py --exp mad_a_iter/gpt/iter3
  python src/get_metrics.py --all

출력: stats/metrics_all.json
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from scipy.stats import wasserstein_distance
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

SCORE_KEYS = ["domain_match", "score_rationale_consistency", "specificity", "groundedness"]
AREAS = ["content", "organization", "expression"]
MODELS = ["gemma", "qwen", "llama", "gpt"]

EXP_REGISTRY: Dict[str, tuple[str, bool]] = {
    "mad_c": ("judge_results/mad_c", False),
}

SINGLE_JUDGE_BASE = Path("judge_results/single_judge")
MAD_A_ITER_BASE = Path("judge_results/mad_a_iter")


def _register_single_judge_exps() -> None:
    if not SINGLE_JUDGE_BASE.exists():
        return
    for judge_dir in sorted(SINGLE_JUDGE_BASE.iterdir()):
        if judge_dir.is_dir():
            key = f"single_judge/{judge_dir.name}"
            EXP_REGISTRY[key] = (str(judge_dir), False)


def _register_mad_a_iter_exps() -> None:
    if not MAD_A_ITER_BASE.exists():
        return
    for judge_dir in sorted(MAD_A_ITER_BASE.iterdir()):
        if not judge_dir.is_dir():
            continue
        for iter_dir in sorted(judge_dir.iterdir()):
            if iter_dir.is_dir():
                key = f"mad_a_iter/{judge_dir.name}/{iter_dir.name}"
                EXP_REGISTRY[key] = (str(iter_dir), True)


def _register_supported_exps() -> None:
    """outline.md 기준 실험만 동적으로 등록."""
    _register_single_judge_exps()
    _register_mad_a_iter_exps()


def collect_scores_from_files(files: List[Path], uses_final: bool) -> Dict[str, List[float]]:
    """score_key → [float, ...] 누적. 3개 area 풀링."""
    accum: Dict[str, List[float]] = {k: [] for k in SCORE_KEYS}

    for p in files:
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARNING] 읽기 실패 {p}: {e}")
            continue

        records = data if isinstance(data, list) else [data]
        for rec in records:
            for area in AREAS:
                area_block = rec.get("judge", {}).get(area, {})
                if not isinstance(area_block, dict):
                    continue
                score_block = area_block.get("final", {}) if uses_final else area_block
                if not isinstance(score_block, dict):
                    continue
                for key in SCORE_KEYS:
                    val = score_block.get(key)
                    if val is not None:
                        try:
                            accum[key].append(float(val))
                        except (TypeError, ValueError):
                            pass

    return accum


def compute_metrics(scores: List[float]) -> Optional[Dict[str, Any]]:
    if not scores:
        return None

    n = len(scores)
    mean = sum(scores) / n
    variance = sum((x - mean) ** 2 for x in scores) / n
    std = math.sqrt(variance)

    sorted_s = sorted(scores)
    q1 = sorted_s[int(n * 0.25)]
    q3 = sorted_s[int(n * 0.75)]
    iqr = q3 - q1

    score_range = max(scores) - min(scores)
    cv = std / mean if mean != 0 else 0.0

    counts = Counter(round(s) for s in scores)
    probs = [c / n for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    n_entropy = entropy / math.log2(5) if math.log2(5) > 0 else 0.0
    mode_freq = max(counts.values()) / n if counts else 0.0

    result = {
        "count": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "range": round(score_range, 4),
        "iqr": round(iqr, 4),
        "cv": round(cv, 4),
        "normalized_entropy": round(n_entropy, 4),
        "mode_freq": round(mode_freq, 4),
    }

    if HAS_SCIPY:
        uniform = [1 / 5] * 5
        actual = [counts.get(i, 0) / n for i in range(1, 6)]
        wd = wasserstein_distance(range(1, 6), range(1, 6), actual, uniform)
        result["wasserstein_from_uniform"] = round(wd, 4)

    return result


def aggregate_exp(exp_key: str) -> Optional[Dict[str, Any]]:
    if exp_key not in EXP_REGISTRY:
        print(f"[WARNING] 등록되지 않은 실험: {exp_key}")
        return None

    base_dir_str, uses_final = EXP_REGISTRY[exp_key]
    base_dir = Path(base_dir_str)
    if not base_dir.exists():
        print(f"[WARNING] 디렉토리 없음: {base_dir}")
        return None

    exp_result: Dict[str, Any] = {}
    for model in MODELS:
        model_dir = base_dir / model
        if not model_dir.exists():
            continue

        files = sorted(model_dir.glob("*.json"))
        if not files:
            continue

        print(f"[INFO] 처리 중: {exp_key}/{model}")
        accum = collect_scores_from_files(files, uses_final)
        model_result: Dict[str, Any] = {}
        for key in SCORE_KEYS:
            model_result[key] = compute_metrics(accum[key])
        exp_result[model] = model_result

    return exp_result if exp_result else None


def main():
    _register_supported_exps()

    parser = argparse.ArgumentParser(description="outline.md 기준 Range + Distribution 메트릭 집계")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exp", help="단일 실험 키 (예: single_judge/gpt, mad_c, mad_a_iter/gpt/iter3)")
    group.add_argument("--all", action="store_true", help="등록된 모든 실험 집계")
    parser.add_argument("--output", default="stats/metrics_all.json", help="출력 파일 경로")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    existing: Dict[str, Any] = {}
    if output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass

    if args.all:
        all_results: Dict[str, Any] = {}
        for exp_key in EXP_REGISTRY:
            result = aggregate_exp(exp_key)
            if result:
                all_results[exp_key] = result
                print(f"[DONE] {exp_key}: {list(result.keys())}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
    else:
        result = aggregate_exp(args.exp)
        if result:
            existing[args.exp] = result
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            print(f"[DONE] {args.exp}: {list(result.keys())}")
        else:
            print(f"[ERROR] 집계 실패: {args.exp}")
            return

    print(f"\n[INFO] 저장 완료: {output_path}")

    final_data = existing if not args.all else {}
    if args.all:
        try:
            with open(output_path, encoding="utf-8") as f:
                final_data = json.load(f)
        except Exception:
            pass

    for exp_key, exp_data in final_data.items():
        print(f"\n=== {exp_key} ===")
        for model, model_data in exp_data.items():
            print(f"  [{model}]")
            for score_key, metrics in model_data.items():
                if metrics:
                    print(
                        f"    {score_key}: mean={metrics['mean']:.3f}, std={metrics['std']:.3f}, "
                        f"cv={metrics['cv']:.3f}, n_entropy={metrics['normalized_entropy']:.3f}, "
                        f"mode_freq={metrics['mode_freq']:.3f}"
                    )


if __name__ == "__main__":
    main()
