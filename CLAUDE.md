# CLAUDE.md

This file gives coding agents the current project map. Keep it aligned with `outline.md`, `plan.md`, and `docs/`.

## Environment

- Current workspace: `/home/msong/MAD-on-Judge`
- Python entrypoint style: run scripts from the repo root with `python` or `python3`.
- Local LM Studio endpoint used by local judge models: `http://localhost:1234/v1`
- GPT judge scripts require an OpenAI API key through the script config or environment, depending on the script.

## Project Overview

This repo studies whether multi-agent debate judges reduce score diversity in subjective qualitative evaluation tasks.

Two domains are used:

- Korean essay scoring: judge models evaluate the quality of model-produced `predicted_score + rationale`.
- SummEval summarization: judge models evaluate article-summary pairs on English summarization criteria.

Main research questions:

- RQ1: score distribution differences across `single`, `mad1`, `mad2 iter3`, `mad2 iter5`
- RQ2: oscillation, flip, and convergence behavior inside mad2 rounds
- RQ3: whether mad2 convergence is caused by logical persuasion or numeric anchoring

## Directory Structure

```text
.
├── data/
│   ├── NIKL_GRADING WRITING DATA 2024/       raw NIKL essay data
│   └── selected_prompt_jsons_100/            current 600-sample essay set
├── docs/                                     usage docs
├── src/
│   ├── env_utils.py
│   ├── prompts/
│   ├── essay/
│   └── summeval/
├── stats/                                    aggregate tables and markdown summaries
├── summeval_judge_input/                     SummEval normalized judge inputs
├── summeval_judge_results/                   SummEval experiment outputs
└── 실험결과 시각화/
```

Generated caches and raw debug responses should not be committed:

- `__pycache__/`
- `*.pyc`
- `debug/raw_responses/`
- `lagacy/`

## Current Naming

| Paper term | Current script/path term |
|---|---|
| single | `single_judge` |
| mad1 | `mad1_critic_defender`, `mad1` |
| mad2 | `mad2_consensus_iter`, `mad2_iter` |
| text-only ablation | `mad2_text_only` |

Legacy names still present in old outputs:

- `mad_c` means old mad1-style critic/defender output.
- `mad_a_iter` means old mad2 iterative output.

New code and new docs should use `mad1` and `mad2_iter`.

## Essay Pipeline

### Data Preparation

- `src/essay/prepare_nikl_essay_dataset.py`
- Default output: `data/selected_prompt_jsons_100/`
- Some judge scripts still default to `data/selected_prompt_jsons/`; align this before large runs.

```bash
python src/essay/prepare_nikl_essay_dataset.py
```

### Essay Scoring

Model scoring scripts:

- `src/essay/inference_essay/single_gpt.py`
- `src/essay/inference_essay/single_ollama.py`
- `src/essay/inference_essay/single_lmstudio.py`

Expected result layout:

```text
inference_results/{gemma,qwen,llama,gpt}/
```

### Essay Judges

| Purpose | Script | Output |
|---|---|---|
| single baseline | `src/essay/legacy_rationale_single_judge.py` | `judge_results/single_judge/{judge_model}/{essay_model}/` |
| mad1 critic-defender | `src/essay/mad1_critic_defender.py` | `judge_results/mad1/{essay_model}/` |
| mad2 iterative consensus | `src/essay/mad2_consensus_iter.py` | `judge_results/mad2_iter/{judge_model}/iter{N}/{essay_model}/` |
| mad2 text-only ablation | `src/essay/mad2_text_only.py` | `judge_results/mad2_text_only/{judge_model}/iter{N}/{essay_model}/` |

Example commands:

```bash
python src/essay/legacy_rationale_single_judge.py --judge-model gpt
python src/essay/mad2_consensus_iter.py --judge-model gpt --model gpt --iterations 3
python src/essay/mad2_text_only.py --judge-model gpt --model gpt --iterations 5
```

### Essay Analysis

| Purpose | Script | Output |
|---|---|---|
| direct score distribution | `src/essay/rq1_score_distribution.py` | `stats/essay_direct_distribution_metrics.json` |
| mad2 oscillation | `src/essay/rq2_mad2_oscillation.py` | `stats/rq2_oscillation_{judge_model}.json/.md` if custom output is supplied |
| RQ3 anchoring | `src/essay/analysis/rq3_anchoring.py` | `stats/rq3_anchoring_gpt_iter5.json/.md` |
| human average scoring metrics | `src/essay/essay_scoring_performance.py` | `stats/essay_scoring_metrics.json` |

## SummEval Pipeline

SummEval uses `coherence`, `consistency`, `fluency`, `relevance`.

| Purpose | Script | Output |
|---|---|---|
| prepare inputs | `src/summeval/prepare_summeval.py` | `summeval_judge_input/{system_name}/` |
| single baseline | `src/summeval/single_judge_summeval.py` | `summeval_judge_results/single_judge/{judge_model}/{system_name}/` |
| mad1 critic-defender | `src/summeval/mad1_critic_defender_summeval.py` | `summeval_judge_results/mad1/{judge_model}/{system_name}/` |
| mad2 iterative consensus | `src/summeval/mad2_consensus_iter_summeval.py` | `summeval_judge_results/mad2_iter/{judge_model}/iter{N}/{system_name}/` |
| RQ1 distribution | `src/summeval/rq1_score_distribution_summeval.py` | `stats/summeval_metrics_all.json` |

Recommended order:

```bash
python src/summeval/prepare_summeval.py --sample-size 100 --seed 42
python src/summeval/single_judge_summeval.py --judge-model gpt --workers 8
python src/summeval/mad1_critic_defender_summeval.py --judge-model gpt --workers 8
python src/summeval/mad2_consensus_iter_summeval.py --judge-model gpt --iterations 3 --workers 8
python src/summeval/mad2_consensus_iter_summeval.py --judge-model gpt --iterations 5 --workers 8
python src/summeval/rq1_score_distribution_summeval.py --all
```

## Docs

Use `docs/README.md` as the entry point. When changing script arguments, default paths, output schema, or experiment naming, update the matching docs file in the same change.
