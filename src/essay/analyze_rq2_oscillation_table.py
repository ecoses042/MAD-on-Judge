"""
RQ2 (합의 MAD 진동 현상 분석) 요약 표 생성 스크립트.

기본 대상:
  judge_results/mad_a_iter/gpt/iter*/{essay_model}/*.json

핵심 지표 (outline.md 기준):
  - flip_count
  - delta_strict_r
  - delta_lenient_r
  - convergence_final

출력:
  - 콘솔 표
  - Markdown 표 파일
  - JSON 요약 파일

예시:
  python src/analyze_rq2_oscillation_table.py
  python src/analyze_rq2_oscillation_table.py --judge-model gpt --iterations 3 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


AREAS = ["content", "organization", "expression"]
ESSAY_MODELS = ["gemma", "qwen", "llama", "gpt"]
DEFAULT_OUTPUT_MD = "stats/rq2_oscillation_gpt.md"
DEFAULT_OUTPUT_JSON = "stats/rq2_oscillation_gpt.json"


def sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def round_sort_key(round_name: str) -> int:
    try:
        return int(round_name.split("_")[1])
    except (IndexError, ValueError):
        return 9999


def load_records(file_path: Path) -> list[dict[str, Any]]:
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def extract_round_series(area_block: dict[str, Any]) -> tuple[list[float], list[float]]:
    round_names = sorted(
        [k for k in area_block.keys() if k.startswith("round_")],
        key=round_sort_key,
    )
    strict_scores: list[float] = []
    lenient_scores: list[float] = []

    for round_name in round_names:
        round_block = area_block.get(round_name, {})
        strict_val = round_block.get("strict", {}).get("overall_judge")
        lenient_val = round_block.get("lenient", {}).get("overall_judge")
        if strict_val is None or lenient_val is None:
            continue
        strict_scores.append(float(strict_val))
        lenient_scores.append(float(lenient_val))

    return strict_scores, lenient_scores


def analyze_area_block(area_block: dict[str, Any]) -> dict[str, float] | None:
    strict_scores, lenient_scores = extract_round_series(area_block)
    if len(strict_scores) < 2 or len(lenient_scores) < 2:
        return None

    strict_abs_deltas = [
        abs(strict_scores[i] - strict_scores[i - 1]) for i in range(1, len(strict_scores))
    ]
    lenient_abs_deltas = [
        abs(lenient_scores[i] - lenient_scores[i - 1]) for i in range(1, len(lenient_scores))
    ]
    strict_signed_deltas = [
        strict_scores[i] - strict_scores[i - 1] for i in range(1, len(strict_scores))
    ]
    lenient_signed_deltas = [
        lenient_scores[i] - lenient_scores[i - 1] for i in range(1, len(lenient_scores))
    ]

    gaps = [s - l for s, l in zip(strict_scores, lenient_scores)]
    flip_count = 0
    for i in range(1, len(gaps)):
        if sign(gaps[i]) != sign(gaps[i - 1]):
            flip_count += 1

    convergence_final = abs(strict_scores[-1] - lenient_scores[-1])

    return {
        "flip_count": float(flip_count),
        "delta_strict_abs_mean": mean(strict_abs_deltas),
        "delta_lenient_abs_mean": mean(lenient_abs_deltas),
        "delta_strict_signed_mean": mean(strict_signed_deltas),
        "delta_lenient_signed_mean": mean(lenient_signed_deltas),
        "convergence_final": convergence_final,
    }


def summarize_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "n": float(len(metric_rows)),
        "flip_count_mean": mean(row["flip_count"] for row in metric_rows),
        "flip_rate": mean(1.0 if row["flip_count"] > 0 else 0.0 for row in metric_rows),
        "delta_strict_abs_mean": mean(row["delta_strict_abs_mean"] for row in metric_rows),
        "delta_lenient_abs_mean": mean(row["delta_lenient_abs_mean"] for row in metric_rows),
        "delta_strict_signed_mean": mean(row["delta_strict_signed_mean"] for row in metric_rows),
        "delta_lenient_signed_mean": mean(row["delta_lenient_signed_mean"] for row in metric_rows),
        "convergence_final_mean": mean(row["convergence_final"] for row in metric_rows),
    }


def collect_iteration_summary(base_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": {}, "by_essay_model": {}, "by_area": {}}
    overall_rows: list[dict[str, float]] = []
    by_model_rows: dict[str, list[dict[str, float]]] = {m: [] for m in ESSAY_MODELS}
    by_area_rows: dict[str, list[dict[str, float]]] = {a: [] for a in AREAS}

    for essay_model in ESSAY_MODELS:
        model_dir = base_dir / essay_model
        if not model_dir.exists():
            continue

        for file_path in sorted(model_dir.glob("*.json")):
            try:
                records = load_records(file_path)
            except Exception as e:
                print(f"[WARNING] 읽기 실패 {file_path}: {e}")
                continue

            for record in records:
                judge_block = record.get("judge", {})
                for area in AREAS:
                    area_block = judge_block.get(area)
                    if not isinstance(area_block, dict):
                        continue
                    row = analyze_area_block(area_block)
                    if row is None:
                        continue
                    overall_rows.append(row)
                    by_model_rows[essay_model].append(row)
                    by_area_rows[area].append(row)

    if overall_rows:
        summary["overall"] = summarize_metrics(overall_rows)

    for essay_model, rows in by_model_rows.items():
        if rows:
            summary["by_essay_model"][essay_model] = summarize_metrics(rows)

    for area, rows in by_area_rows.items():
        if rows:
            summary["by_area"][area] = summarize_metrics(rows)

    return summary


def _fmt(value: float, digits: int = 3) -> str:
    if digits == 0:
        return str(int(round(value)))
    return f"{value:.{digits}f}"


def build_markdown(summary_by_iter: dict[str, Any], judge_model: str) -> str:
    lines: list[str] = []
    lines.append(f"# RQ2 Oscillation Summary ({judge_model} judge)")
    lines.append("")

    for iter_name, summary in summary_by_iter.items():
        lines.append(f"## {iter_name}")
        lines.append("")
        lines.append("### Overall")
        lines.append("")
        lines.append(
            "| scope | n | flip_count_mean | flip_rate | delta_strict_abs | "
            "delta_lenient_abs | delta_strict_signed | delta_lenient_signed | convergence_final |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        overall = summary.get("overall", {})
        if overall:
            lines.append(
                "| overall | "
                f"{_fmt(overall['n'], 0)} | "
                f"{_fmt(overall['flip_count_mean'])} | "
                f"{_fmt(overall['flip_rate'])} | "
                f"{_fmt(overall['delta_strict_abs_mean'])} | "
                f"{_fmt(overall['delta_lenient_abs_mean'])} | "
                f"{_fmt(overall['delta_strict_signed_mean'])} | "
                f"{_fmt(overall['delta_lenient_signed_mean'])} | "
                f"{_fmt(overall['convergence_final_mean'])} |"
            )
        else:
            lines.append("| overall | 0 | - | - | - | - | - | - | - |")

        lines.append("")
        lines.append("### By Essay Model")
        lines.append("")
        lines.append(
            "| essay_model | n | flip_count_mean | flip_rate | delta_strict_abs | "
            "delta_lenient_abs | delta_strict_signed | delta_lenient_signed | convergence_final |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for essay_model in ESSAY_MODELS:
            row = summary.get("by_essay_model", {}).get(essay_model)
            if not row:
                continue
            lines.append(
                f"| {essay_model} | "
                f"{_fmt(row['n'], 0)} | "
                f"{_fmt(row['flip_count_mean'])} | "
                f"{_fmt(row['flip_rate'])} | "
                f"{_fmt(row['delta_strict_abs_mean'])} | "
                f"{_fmt(row['delta_lenient_abs_mean'])} | "
                f"{_fmt(row['delta_strict_signed_mean'])} | "
                f"{_fmt(row['delta_lenient_signed_mean'])} | "
                f"{_fmt(row['convergence_final_mean'])} |"
            )

        lines.append("")
        lines.append("### By Area")
        lines.append("")
        lines.append(
            "| area | n | flip_count_mean | flip_rate | delta_strict_abs | "
            "delta_lenient_abs | delta_strict_signed | delta_lenient_signed | convergence_final |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for area in AREAS:
            row = summary.get("by_area", {}).get(area)
            if not row:
                continue
            lines.append(
                f"| {area} | "
                f"{_fmt(row['n'], 0)} | "
                f"{_fmt(row['flip_count_mean'])} | "
                f"{_fmt(row['flip_rate'])} | "
                f"{_fmt(row['delta_strict_abs_mean'])} | "
                f"{_fmt(row['delta_lenient_abs_mean'])} | "
                f"{_fmt(row['delta_strict_signed_mean'])} | "
                f"{_fmt(row['delta_lenient_signed_mean'])} | "
                f"{_fmt(row['convergence_final_mean'])} |"
            )
        lines.append("")

    return "\n".join(lines)


def print_console_tables(summary_by_iter: dict[str, Any]) -> None:
    for iter_name, summary in summary_by_iter.items():
        print(f"\n=== {iter_name} / overall ===")
        overall = summary.get("overall", {})
        if not overall:
            print("데이터 없음")
            continue
        print(
            "n={n:.0f}, flip_count_mean={flip:.3f}, flip_rate={rate:.3f}, "
            "delta_strict_abs={ds_abs:.3f}, delta_lenient_abs={dl_abs:.3f}, "
            "delta_strict_signed={ds_sig:.3f}, delta_lenient_signed={dl_sig:.3f}, "
            "convergence_final={conv:.3f}".format(
                n=overall["n"],
                flip=overall["flip_count_mean"],
                rate=overall["flip_rate"],
                ds_abs=overall["delta_strict_abs_mean"],
                dl_abs=overall["delta_lenient_abs_mean"],
                ds_sig=overall["delta_strict_signed_mean"],
                dl_sig=overall["delta_lenient_signed_mean"],
                conv=overall["convergence_final_mean"],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2 진동 현상 요약 표 생성")
    parser.add_argument("--judge-model", default="gpt", help="judge model 이름 (기본: gpt)")
    parser.add_argument(
        "--iterations",
        nargs="+",
        type=int,
        default=[3, 5],
        help="분석할 iteration 목록 (기본: 3 5)",
    )
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD, help="Markdown 출력 경로")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON, help="JSON 출력 경로")
    args = parser.parse_args()

    summary_by_iter: dict[str, Any] = {}
    for iteration in args.iterations:
        iter_name = f"iter{iteration}"
        base_dir = Path("judge_results") / "mad_a_iter" / args.judge_model / iter_name
        if not base_dir.exists():
            print(f"[WARNING] 디렉토리 없음: {base_dir}")
            continue
        summary_by_iter[iter_name] = collect_iteration_summary(base_dir)

    if not summary_by_iter:
        print("[ERROR] 집계할 데이터가 없습니다.")
        return

    output_md = Path(args.output_md)
    output_json = Path(args.output_json)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    markdown = build_markdown(summary_by_iter, args.judge_model)
    output_md.write_text(markdown, encoding="utf-8")
    output_json.write_text(json.dumps(summary_by_iter, ensure_ascii=False, indent=2), encoding="utf-8")

    print_console_tables(summary_by_iter)
    print(f"\n[INFO] Markdown 저장: {output_md}")
    print(f"[INFO] JSON 저장: {output_json}")


if __name__ == "__main__":
    main()
