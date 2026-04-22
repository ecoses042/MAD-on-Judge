# SummEval Pipeline Usage

## Overview

This pipeline validates domain generalization on the English summarization benchmark `mteb/summeval`.
The main question is whether the MAD-style oscillation and anchoring effects observed in the Korean essay judge pipeline also reproduce on English summarization quality evaluation.

The SummEval pipeline is independent from the essay pipeline:

- It evaluates article-summary pairs rather than essay/domain predictions.
- It uses four summarization criteria: `coherence`, `consistency`, `fluency`, `relevance`.
- It uses English prompts and article-conditioned judging.
- It parallelizes per article-summary item with `ThreadPoolExecutor`.

## Data Flow

Execution flow:

`prepare_summeval.py` -> `summeval_judge_input/` -> `{single_judge_summeval.py, MAD_C_summeval.py, MAD_A_iter_summeval.py}` -> `summeval_judge_results/` -> `get_metrics_summeval.py` -> `stats/summeval_metrics_all.json`

1. `prepare_summeval.py` downloads and samples `mteb/summeval`, then writes one JSON file per `(article, system summary)` pair.
2. Judge scripts read those files and write experiment outputs under `summeval_judge_results/`.
3. `get_metrics_summeval.py` aggregates score distributions across all items for each experiment.

## Directory Layout

```text
summeval_judge_input/
├── system_name_a/
│   ├── article_001.json
│   └── article_002.json
└── system_name_b/
    └── article_001.json

summeval_judge_results/
├── single_judge/
│   └── {judge_model}/
│       └── {system_name}/
│           └── {article_id}.json
├── mad_c/
│   └── {judge_model}/
│       └── {system_name}/
│           └── {article_id}.json
└── mad_a_iter/
    └── {judge_model}/
        └── iter{N}/
            └── {system_name}/
                └── {article_id}.json

stats/
└── summeval_metrics_all.json
```

Input file schema:

```json
{
  "article_id": "...",
  "system_name": "...",
  "article_text": "...",
  "summary_text": "...",
  "gold": {
    "coherence": 3.5,
    "consistency": 4.0,
    "fluency": 4.5,
    "relevance": 3.0
  }
}
```

## Dependencies

Install the required packages from the project root:

```bash
pip install datasets openai scipy
```

`scipy` is optional but recommended because `get_metrics_summeval.py` adds `wasserstein_from_uniform` when SciPy is available.

For GPT-based judging, set `OPENAI_API_KEY`.
For local judging, LM Studio should expose an OpenAI-compatible API at `http://localhost:1234/v1`.

## CLI Examples

Prepare sampled SummEval inputs:

```bash
python src/summeval/prepare_summeval.py --sample-size 100 --seed 42
python src/summeval/prepare_summeval.py --sample-size 200 --output-dir summeval_judge_input
```

Run single-judge evaluation:

```bash
python src/summeval/single_judge_summeval.py --judge-model gpt --workers 8
python src/summeval/single_judge_summeval.py --judge-model gemma --workers 4
```

Run MAD-C evaluation:

```bash
python src/summeval/MAD_C_summeval.py --judge-model gpt --workers 8
python src/summeval/MAD_C_summeval.py --judge-model qwen --workers 4
```

Run iterative MAD-A evaluation:

```bash
python src/summeval/MAD_A_iter_summeval.py --judge-model gpt --iterations 3 --workers 8
python src/summeval/MAD_A_iter_summeval.py --judge-model gpt --iterations 5 --workers 8
python src/summeval/MAD_A_iter_summeval.py --judge-model gemma --iterations 3 --workers 4
```

Aggregate one experiment:

```bash
python src/summeval/get_metrics_summeval.py --exp single_judge/gpt
python src/summeval/get_metrics_summeval.py --exp mad_c/gpt
python src/summeval/get_metrics_summeval.py --exp mad_a_iter/gpt/iter3
```

Aggregate all discovered experiments:

```bash
python src/summeval/get_metrics_summeval.py --all
```

## Key Differences From The Korean Essay Pipeline

- No domain loop: each input is a single article-summary pair, not one essay with multiple `content/organization/expression` domains.
- New criteria: `coherence`, `consistency`, `fluency`, `relevance`.
- English prompts: all judge prompts are written for English summarization evaluation.
- Different data source: `mteb/summeval` from Hugging Face instead of local essay JSON files.
- Parallel unit of work: `ThreadPoolExecutor` processes independent article-summary items, grouped by `system_name`.
- Result structure differs slightly:
  - `single_judge` stores scores in `judge`
  - `mad_c` stores `critic`, `defender`, and `final` under `judge`
  - `mad_a_iter` stores `round_0 ... round_N` and `final` under `judge`

## Recommended Execution Order

1. Prepare the evaluation input files.

```bash
python src/summeval/prepare_summeval.py --sample-size 100 --seed 42
```

2. Run the baseline single judge.

```bash
python src/summeval/single_judge_summeval.py --judge-model gpt --workers 8
```

3. Run MAD-C.

```bash
python src/summeval/MAD_C_summeval.py --judge-model gpt --workers 8
```

4. Run MAD-A iterative settings, usually `iter3` then `iter5`.

```bash
python src/summeval/MAD_A_iter_summeval.py --judge-model gpt --iterations 3 --workers 8
python src/summeval/MAD_A_iter_summeval.py --judge-model gpt --iterations 5 --workers 8
```

5. Aggregate metrics.

```bash
python src/summeval/get_metrics_summeval.py --all
```

Final metrics will be written to `stats/summeval_metrics_all.json`.
