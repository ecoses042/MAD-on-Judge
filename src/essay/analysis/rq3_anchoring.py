"""
RQ3 분석: 합의 MAD의 진동이 점수 숫자에 반응하는지, 근거 내용에 반응하는지 구분한다.

분석 1. adjustment_notes 텍스트에서 숫자/점수 참조(anchor)와
        근거/텍스트 참조(logic) 패턴의 빈도를 집계한다.
분석 2. full 조건과 text-only 조건의 flip/convergence를 비교한다.

기본 비교 범위:
  - full:      judge_results/mad2_iter/gpt/iter5/{gemma,gpt,llama}
  - text-only: judge_results/mad2_text_only/gpt/iter5/{gemma,gpt,llama}

출력:
  - stats/rq3_anchoring_gpt_iter5.json
  - stats/rq3_anchoring_gpt_iter5.md

예시:
  python3 src/essay/analysis/rq3_anchoring.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


AREAS = ["content", "organization", "expression"]
ESSAY_MODELS = ["gemma", "gpt", "llama"]
SIDES = ["strict", "lenient"]
ROUND_PREFIX = "round_"

DEFAULT_FULL_BASE = Path("judge_results/mad2_iter/gpt/iter5")
DEFAULT_TEXT_ONLY_BASE = Path("judge_results/mad2_text_only/gpt/iter5")
DEFAULT_OUTPUT_JSON = Path("stats/rq3_anchoring_gpt_iter5.json")
DEFAULT_OUTPUT_MD = Path("stats/rq3_anchoring_gpt_iter5.md")

ANCHOR_PATTERNS = [
    r"\b[1-5](?:\.0)?\s*점\b",
    r"\b[1-5](?:\.[0-9])?\b",
    r"점수",
    r"overall_judge",
    r"상향\s*조정",
    r"하향\s*조정",
    r"조정",
    r"유지",
    r"높은\s*점수",
    r"낮은\s*점수",
]

LOGIC_PATTERNS = [
    r"근거",
    r"rationale",
    r"essay_text",
    r"텍스트",
    r"문장",
    r"문단",
    r"표현",
    r"주장",
    r"내용",
    r"인용",
    r"언급",
    r"구체적",
    r"사례",
    r"예시",
    r"논리",
]


def round_sort_key(round_name: str) -> int:
    try:
        return int(round_name.split("_")[1])
    except (IndexError, ValueError):
        return 9999


def sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def load_json_records(file_path: Path) -> list[dict[str, Any]]:
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def find_common_files(full_base: Path, text_only_base: Path, essay_model: str) -> list[str]:
    full_files = {p.name for p in (full_base / essay_model).glob("*.json")}
    text_files = {p.name for p in (text_only_base / essay_model).glob("*.json")}
    return sorted(full_files & text_files)


def count_pattern_hits(text: str, patterns: list[str]) -> int:
    return sum(len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in patterns)


def classify_note(text: str) -> dict[str, Any]:
    anchor_hits = count_pattern_hits(text, ANCHOR_PATTERNS)
    logic_hits = count_pattern_hits(text, LOGIC_PATTERNS)

    if anchor_hits > 0 and logic_hits > 0:
        label = "mixed"
    elif anchor_hits > 0:
        label = "anchor_only"
    elif logic_hits > 0:
        label = "logic_only"
    else:
        label = "neither"

    return {
        "anchor_hits": anchor_hits,
        "logic_hits": logic_hits,
        "label": label,
        "is_anchor_dominant": anchor_hits > logic_hits,
        "is_logic_dominant": logic_hits > anchor_hits,
    }


def extract_round_names(area_block: dict[str, Any]) -> list[str]:
    return sorted(
        [key for key in area_block if key.startswith(ROUND_PREFIX)],
        key=round_sort_key,
    )


def extract_overall_series(area_block: dict[str, Any]) -> tuple[list[float], list[float]]:
    strict_scores: list[float] = []
    lenient_scores: list[float] = []

    for round_name in extract_round_names(area_block):
        round_block = area_block.get(round_name, {})
        strict_val = round_block.get("strict", {}).get("overall_judge")
        lenient_val = round_block.get("lenient", {}).get("overall_judge")
        if strict_val is None or lenient_val is None:
            continue
        strict_scores.append(float(strict_val))
        lenient_scores.append(float(lenient_val))

    return strict_scores, lenient_scores


def compute_flip_metrics(area_block: dict[str, Any]) -> dict[str, float] | None:
    strict_scores, lenient_scores = extract_overall_series(area_block)
    if len(strict_scores) < 2:
        return None

    gaps = [s - l for s, l in zip(strict_scores, lenient_scores)]
    flip_count = 0
    for idx in range(1, len(gaps)):
        if sign(gaps[idx]) != sign(gaps[idx - 1]):
            flip_count += 1

    return {
        "flip_count": float(flip_count),
        "flip_rate_binary": 1.0 if flip_count > 0 else 0.0,
        "convergence_final": abs(strict_scores[-1] - lenient_scores[-1]),
        "initial_gap": abs(strict_scores[0] - lenient_scores[0]),
        "gap_reduction": abs(strict_scores[0] - lenient_scores[0]) - abs(strict_scores[-1] - lenient_scores[-1]),
    }


def summarize_note_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_counter = Counter(row["label"] for row in rows)
    return {
        "n_notes": len(rows),
        "anchor_hits_total": sum(row["anchor_hits"] for row in rows),
        "logic_hits_total": sum(row["logic_hits"] for row in rows),
        "anchor_hits_mean": mean(row["anchor_hits"] for row in rows),
        "logic_hits_mean": mean(row["logic_hits"] for row in rows),
        "anchor_dominant_ratio": mean(1.0 if row["is_anchor_dominant"] else 0.0 for row in rows),
        "logic_dominant_ratio": mean(1.0 if row["is_logic_dominant"] else 0.0 for row in rows),
        "label_distribution": dict(label_counter),
    }


def summarize_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "n": float(len(rows)),
        "flip_count_mean": mean(row["flip_count"] for row in rows),
        "flip_rate": mean(row["flip_rate_binary"] for row in rows),
        "convergence_final_mean": mean(row["convergence_final"] for row in rows),
        "initial_gap_mean": mean(row["initial_gap"] for row in rows),
        "gap_reduction_mean": mean(row["gap_reduction"] for row in rows),
    }


def collect_note_analysis(base_dir: Path, essay_models: list[str]) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_area: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_side: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for essay_model in essay_models:
        for file_path in sorted((base_dir / essay_model).glob("*.json")):
            for record in load_json_records(file_path):
                for area in AREAS:
                    area_block = record.get("judge", {}).get(area)
                    if not isinstance(area_block, dict):
                        continue
                    for round_name in extract_round_names(area_block):
                        if round_name == "round_0":
                            continue
                        round_block = area_block.get(round_name, {})
                        for side in SIDES:
                            side_block = round_block.get(side, {})
                            note = side_block.get("adjustment_notes")
                            if not isinstance(note, str) or not note.strip():
                                continue
                            classified = classify_note(note)
                            row = {
                                "essay_model": essay_model,
                                "area": area,
                                "side": side,
                                "round_name": round_name,
                                **classified,
                            }
                            all_rows.append(row)
                            by_model[essay_model].append(row)
                            by_area[area].append(row)
                            by_side[side].append(row)

    summary = {"overall": {}, "by_model": {}, "by_area": {}, "by_side": {}}
    if all_rows:
        summary["overall"] = summarize_note_rows(all_rows)
    for key, rows in by_model.items():
        summary["by_model"][key] = summarize_note_rows(rows)
    for key, rows in by_area.items():
        summary["by_area"][key] = summarize_note_rows(rows)
    for key, rows in by_side.items():
        summary["by_side"][key] = summarize_note_rows(rows)
    return summary


def collect_condition_metrics(base_dir: Path, essay_models: list[str], files_filter: dict[str, list[str]] | None = None) -> dict[str, Any]:
    all_rows: list[dict[str, float]] = []
    by_model: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_area: dict[str, list[dict[str, float]]] = defaultdict(list)

    for essay_model in essay_models:
        file_names = files_filter[essay_model] if files_filter else sorted(p.name for p in (base_dir / essay_model).glob("*.json"))
        for file_name in file_names:
            file_path = base_dir / essay_model / file_name
            if not file_path.exists():
                continue
            for record in load_json_records(file_path):
                for area in AREAS:
                    area_block = record.get("judge", {}).get(area)
                    if not isinstance(area_block, dict):
                        continue
                    metrics = compute_flip_metrics(area_block)
                    if metrics is None:
                        continue
                    all_rows.append(metrics)
                    by_model[essay_model].append(metrics)
                    by_area[area].append(metrics)

    summary = {"overall": {}, "by_model": {}, "by_area": {}}
    if all_rows:
        summary["overall"] = summarize_metric_rows(all_rows)
    for key, rows in by_model.items():
        summary["by_model"][key] = summarize_metric_rows(rows)
    for key, rows in by_area.items():
        summary["by_area"][key] = summarize_metric_rows(rows)
    return summary


def build_comparison(full_summary: dict[str, Any], text_summary: dict[str, Any]) -> dict[str, Any]:
    full_overall = full_summary.get("overall", {})
    text_overall = text_summary.get("overall", {})
    if not full_overall or not text_overall:
        return {}
    return {
        "flip_count_mean_diff": text_overall["flip_count_mean"] - full_overall["flip_count_mean"],
        "flip_rate_diff": text_overall["flip_rate"] - full_overall["flip_rate"],
        "convergence_final_mean_diff": text_overall["convergence_final_mean"] - full_overall["convergence_final_mean"],
        "gap_reduction_mean_diff": text_overall["gap_reduction_mean"] - full_overall["gap_reduction_mean"],
    }


def build_markdown(summary: dict[str, Any]) -> str:
    note_overall = summary["note_analysis"]["overall"]
    full_overall = summary["condition_metrics"]["full"]["overall"]
    text_overall = summary["condition_metrics"]["text_only"]["overall"]
    comparison = summary["condition_comparison"]["overall"]

    lines: list[str] = []
    lines.append("# RQ3 Anchoring Analysis (gpt judge, iter5)")
    lines.append("")
    lines.append("## 1. adjustment_notes 패턴")
    lines.append("")
    lines.append("| n_notes | anchor_hits_total | logic_hits_total | anchor_hits_mean | logic_hits_mean | anchor_dominant_ratio | logic_dominant_ratio | anchor_only | logic_only | mixed | neither |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {note_overall['n_notes']} | "
        f"{note_overall['anchor_hits_total']} | "
        f"{note_overall['logic_hits_total']} | "
        f"{note_overall['anchor_hits_mean']:.3f} | "
        f"{note_overall['logic_hits_mean']:.3f} | "
        f"{note_overall['anchor_dominant_ratio']:.3f} | "
        f"{note_overall['logic_dominant_ratio']:.3f} | "
        f"{note_overall['label_distribution'].get('anchor_only', 0)} | "
        f"{note_overall['label_distribution'].get('logic_only', 0)} | "
        f"{note_overall['label_distribution'].get('mixed', 0)} | "
        f"{note_overall['label_distribution'].get('neither', 0)} |"
    )
    lines.append("")
    lines.append("## 2. Full vs Text-only")
    lines.append("")
    lines.append("| condition | n | flip_count_mean | flip_rate | convergence_final_mean | initial_gap_mean | gap_reduction_mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| full | {int(full_overall['n'])} | {full_overall['flip_count_mean']:.3f} | {full_overall['flip_rate']:.3f} | "
        f"{full_overall['convergence_final_mean']:.3f} | {full_overall['initial_gap_mean']:.3f} | {full_overall['gap_reduction_mean']:.3f} |"
    )
    lines.append(
        f"| text_only | {int(text_overall['n'])} | {text_overall['flip_count_mean']:.3f} | {text_overall['flip_rate']:.3f} | "
        f"{text_overall['convergence_final_mean']:.3f} | {text_overall['initial_gap_mean']:.3f} | {text_overall['gap_reduction_mean']:.3f} |"
    )
    lines.append("")
    lines.append("## 3. Text-only - Full 차이")
    lines.append("")
    lines.append("| flip_count_mean_diff | flip_rate_diff | convergence_final_mean_diff | gap_reduction_mean_diff |")
    lines.append("|---:|---:|---:|---:|")
    lines.append(
        f"| {comparison['flip_count_mean_diff']:.3f} | {comparison['flip_rate_diff']:.3f} | "
        f"{comparison['convergence_final_mean_diff']:.3f} | {comparison['gap_reduction_mean_diff']:.3f} |"
    )
    lines.append("")
    lines.append("## 4. Interpretation")
    lines.append("")
    lines.append(summary["interpretation"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ3 anchoring analysis")
    parser.add_argument("--full-base", type=Path, default=DEFAULT_FULL_BASE)
    parser.add_argument("--text-only-base", type=Path, default=DEFAULT_TEXT_ONLY_BASE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--essay-models", nargs="+", default=ESSAY_MODELS)
    args = parser.parse_args()

    common_files = {
        essay_model: find_common_files(args.full_base, args.text_only_base, essay_model)
        for essay_model in args.essay_models
    }

    note_analysis = collect_note_analysis(args.full_base, args.essay_models)
    full_metrics = collect_condition_metrics(args.full_base, args.essay_models, files_filter=common_files)
    text_only_metrics = collect_condition_metrics(args.text_only_base, args.essay_models, files_filter=common_files)
    comparison = {"overall": build_comparison(full_metrics, text_only_metrics)}

    note_overall = note_analysis.get("overall", {})
    metric_diff = comparison["overall"]

    interpretation_parts: list[str] = []
    if note_overall:
        interpretation_parts.append(
            "adjustment_notes에서는 내용 참조 표현보다 숫자/점수 관련 표현이 더 많이 나타났다."
            if note_overall["anchor_hits_total"] > note_overall["logic_hits_total"]
            else "adjustment_notes에서는 숫자/점수 관련 표현보다 내용 참조 표현이 더 많이 나타났다."
        )
        interpretation_parts.append(
            "앵커링 우세 노트 비율이 내용 우세 노트 비율보다 높다."
            if note_overall["anchor_dominant_ratio"] > note_overall["logic_dominant_ratio"]
            else "내용 우세 노트 비율이 앵커링 우세 노트 비율보다 높다."
        )
    if metric_diff:
        interpretation_parts.append(
            "text-only에서 flip이 감소했다."
            if metric_diff["flip_rate_diff"] < 0
            else "text-only에서도 flip이 줄지 않았고, 진동이 유지되거나 증가했다."
        )
        interpretation_parts.append(
            "따라서 현재 데이터만 보면 숫자 앵커링 단독 가설은 강하게 지지되지 않으며, 텍스트 피드백 자체의 상호 모방/반복도 중요한 원인으로 보인다."
            if metric_diff["flip_rate_diff"] >= 0
            else "따라서 현재 데이터는 숫자 앵커링이 진동의 중요한 원인이라는 가설을 지지한다."
        )

    summary = {
        "settings": {
            "full_base": str(args.full_base),
            "text_only_base": str(args.text_only_base),
            "essay_models": args.essay_models,
            "common_file_counts": {k: len(v) for k, v in common_files.items()},
        },
        "note_analysis": note_analysis,
        "condition_metrics": {
            "full": full_metrics,
            "text_only": text_only_metrics,
        },
        "condition_comparison": comparison,
        "interpretation": " ".join(interpretation_parts),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(build_markdown(summary), encoding="utf-8")

    print(json.dumps(summary["settings"], ensure_ascii=False, indent=2))
    print(summary["interpretation"])


if __name__ == "__main__":
    main()
