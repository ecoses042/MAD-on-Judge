import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # adds src/ to path
from env_utils import load_project_env

load_project_env(__file__)

import argparse
import json


def avg(lst):
    return round(sum(lst) / len(lst), 2) if lst else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="summeval_judge_input")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets library not installed. Run: pip install datasets")
        sys.exit(1)

    print("Loading mteb/summeval from HuggingFace...")
    ds = load_dataset("mteb/summeval", split="test")
    print(f"Dataset features: {ds.features}")
    print(f"Total rows: {len(ds)}")

    sample = ds.shuffle(seed=args.seed).select(range(min(args.sample_size, len(ds))))

    output_base = Path(args.output_dir)
    count = 0

    for row in sample:
        article_id = str(row.get("id", row.get("article_id", f"row_{count}")))
        article_text = row.get("text", row.get("article_text", ""))
        summaries = row.get("machine_summaries", [])
        human_ann = row.get("human_annotations", {})
        sys_names = row.get(
            "sys_summ_ids",
            row.get("system_ids", [f"system_{i}" for i in range(len(summaries))]),
        )

        for i, (sys_name, summary_text) in enumerate(zip(sys_names, summaries)):
            sys_name = str(sys_name)
            gold = {}
            for criterion in ["coherence", "consistency", "fluency", "relevance"]:
                ann_list = human_ann.get(criterion, [])
                if i < len(ann_list):
                    scores_for_system = ann_list[i]
                    if isinstance(scores_for_system, list):
                        gold[criterion] = avg([s for s in scores_for_system if s is not None])
                    else:
                        gold[criterion] = scores_for_system

            out = {
                "article_id": article_id,
                "system_name": sys_name,
                "article_text": article_text,
                "summary_text": summary_text,
                "gold": gold,
            }

            out_dir = output_base / sys_name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{article_id}.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            count += 1

    print(f"Done. Created {count} files across {len(list(output_base.iterdir()))} system directories.")


if __name__ == "__main__":
    main()
