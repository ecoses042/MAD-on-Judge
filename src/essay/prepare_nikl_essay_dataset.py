from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_RAW_DIR = Path("data/NIKL_GRADING WRITING DATA\u00a02024")
DEFAULT_OUTPUT_DIR = Path("data/selected_prompt_jsons_100")
PROMPT_IDS = ("Q4", "Q5", "Q6", "Q7", "Q8", "Q9")
MAX_SCORES = {
    "con": 25.0,
    "org": 10.0,
    "exp": 10.0,
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return data


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_dir_name(name: str) -> str:
    return name.replace("\u00a0", " ").strip().lower()


def find_raw_dir(raw_dir: Path) -> Path:
    if raw_dir.exists():
        return raw_dir

    parent = raw_dir.parent if raw_dir.parent != Path("") else Path(".")
    if not parent.exists():
        raise FileNotFoundError(f"Raw data parent directory does not exist: {parent}")

    desired = normalize_dir_name(raw_dir.name)
    for candidate in parent.iterdir():
        if candidate.is_dir() and normalize_dir_name(candidate.name) == desired:
            return candidate

    raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")


def safe_mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def normalize_to_5_scale(score: float | None, max_score: float) -> float | None:
    if score is None:
        return None
    return (score / max_score) * 5.0


def get_dimension_score(
    evaluation_data: dict[str, Any],
    dimension: str,
    evaluator: int,
) -> float | None:
    block = evaluation_data.get(f"eva_score_{dimension}", {})
    if not isinstance(block, dict):
        return None
    value = block.get(f"evaluator{evaluator}_score_total_{dimension}")
    return float(value) if isinstance(value, (int, float)) else None


def get_5scale_label(
    evaluation_data: dict[str, Any],
    evaluator: int | None,
) -> dict[str, float | None]:
    raw_scores: dict[str, float | None] = {}
    for dimension in MAX_SCORES:
        if evaluator is None:
            values = [
                score
                for score in (
                    get_dimension_score(evaluation_data, dimension, 1),
                    get_dimension_score(evaluation_data, dimension, 2),
                )
                if score is not None
            ]
            raw_scores[dimension] = safe_mean(values)
        else:
            raw_scores[dimension] = get_dimension_score(evaluation_data, dimension, evaluator)

    return {
        dimension: normalize_to_5_scale(score, MAX_SCORES[dimension])
        for dimension, score in raw_scores.items()
    }


def join_paragraphs(paragraphs: list[dict[str, Any]]) -> str:
    texts: list[str] = []
    for paragraph in paragraphs:
        text = str(paragraph.get("form", "")).strip()
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def count_sentences(paragraphs: list[dict[str, Any]]) -> int:
    total = 0
    for paragraph in paragraphs:
        sentences = paragraph.get("sentence", [])
        if isinstance(sentences, list):
            total += len(sentences)
    return total


def transform_document(doc: dict[str, Any], root_metadata: dict[str, Any]) -> dict[str, Any] | None:
    metadata = doc.get("metadata", {})
    prompt = metadata.get("prompt", {}) if isinstance(metadata, dict) else {}
    prompt_id = str(prompt.get("prompt_num", "")).strip()
    if prompt_id not in PROMPT_IDS:
        return None

    paragraphs = doc.get("paragraph", [])
    if not isinstance(paragraphs, list):
        return None

    essay_text = join_paragraphs(paragraphs)
    evaluation = doc.get("evaluation", {})
    evaluation_data = evaluation.get("evaluation_data", {}) if isinstance(evaluation, dict) else {}
    if not essay_text or not isinstance(evaluation_data, dict):
        return None

    written_stat = metadata.get("written_stat", {}) if isinstance(metadata, dict) else {}

    return {
        "essay_id": str(doc.get("id", "")).strip(),
        "source_id": root_metadata.get("title", ""),
        "prompt_id": prompt_id,
        "prompt_text": str(prompt.get("prompt_con", "")).strip(),
        "essay_text": essay_text,
        "meta": {
            "written_length": written_stat.get("written_length"),
            "paragraph_num": written_stat.get("paragraph_num"),
            "sentence_num_original": written_stat.get("sentence_num"),
            "sentence_num_recounted": count_sentences(paragraphs),
        },
        "label_5scale_evaluator1": get_5scale_label(evaluation_data, 1),
        "label_5scale_evaluator2": get_5scale_label(evaluation_data, 2),
        "label_5scale_average": get_5scale_label(evaluation_data, None),
    }


def load_samples(raw_dir: Path) -> dict[str, list[dict[str, Any]]]:
    samples_by_prompt: dict[str, list[dict[str, Any]]] = {prompt_id: [] for prompt_id in PROMPT_IDS}

    for json_file in sorted(raw_dir.glob("*.json")):
        data = load_json(json_file)
        root_metadata = data.get("metadata", {})
        documents = data.get("document", [])
        if not isinstance(root_metadata, dict) or not isinstance(documents, list):
            continue

        for doc in documents:
            if not isinstance(doc, dict):
                continue
            sample = transform_document(doc, root_metadata)
            if sample is not None:
                samples_by_prompt[sample["prompt_id"]].append(sample)

    return samples_by_prompt


def write_prompt_samples(
    samples_by_prompt: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    samples_per_prompt: int,
    seed: int,
) -> dict[str, int]:
    rng = random.Random(seed)
    counts: dict[str, int] = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    for prompt_id in PROMPT_IDS:
        candidates = samples_by_prompt[prompt_id]
        if len(candidates) < samples_per_prompt:
            raise ValueError(
                f"{prompt_id} has only {len(candidates)} valid samples; "
                f"requested {samples_per_prompt}."
            )

        selected = rng.sample(candidates, samples_per_prompt)
        selected.sort(key=lambda item: item["essay_id"])
        counts[prompt_id] = len(selected)

        for index, sample in enumerate(selected, start=1):
            safe_essay_id = sample["essay_id"].replace("/", "_").replace("\\", "_").replace(":", "_")
            output_file = output_dir / f"{prompt_id}_{index:03d}_{safe_essay_id}.json"
            save_json([sample], output_file)

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a random 100-per-topic NIKL essay dataset in selected_prompt_jsons format."
    )
    parser.add_argument("--raw-data-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples-per-prompt", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = find_raw_dir(args.raw_data_dir)

    samples_by_prompt = load_samples(raw_dir)
    counts = write_prompt_samples(
        samples_by_prompt=samples_by_prompt,
        output_dir=args.output_dir,
        samples_per_prompt=args.samples_per_prompt,
        seed=args.seed,
    )

    print(f"raw_data_dir: {raw_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"seed: {args.seed}")
    for prompt_id in PROMPT_IDS:
        print(f"{prompt_id}: {counts[prompt_id]} / {len(samples_by_prompt[prompt_id])}")
    print(f"total: {sum(counts.values())}")


if __name__ == "__main__":
    main()
