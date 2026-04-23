# MAD-on-Judge 실행 계획

> 기준 문서: `outline.md` v0.3  
> 갱신일: 2026-04-23

## 목표

현재 repo는 에세이 파이프라인과 SummEval 파이프라인이 모두 구현되어 있지만, 결과 산출물은 아직 `outline.md`의 RQ1/RQ2/RQ3를 완전히 채우지 못한다. 이 계획은 남은 작업을 실행 가능한 순서로 정리한다.

핵심 산출물은 다음 세 가지다.

1. RQ1: `single / mad1 / mad2 iter5` 점수 분포 비교표
2. RQ2: mad2 round별 flip, delta, convergence 분석표
3. RQ3: full vs text-only ablation 비교표와 숫자 앵커링 해석

## 현재 표준 경로

### 에세이

| 용도 | 경로 |
|---|---|
| 원본 NIKL 데이터 | `data/NIKL_GRADING WRITING DATA 2024/` |
| 100-per-prompt 샘플 | `data/selected_prompt_jsons_100/` |
| 기존 스크립트 기본 에세이 경로 | `data/selected_prompt_jsons/` |
| 모델 채점 결과 | `inference_results/{gemma,qwen,llama,gpt}/` |
| single judge 결과 | `judge_results/single_judge/{judge_model}/{essay_model}/` |
| mad1 결과 | `judge_results/mad1/{essay_model}/` |
| mad2 결과 | `judge_results/mad2_iter/{judge_model}/iter5/{essay_model}/` |
| mad2 text-only 결과 | `judge_results/mad2_text_only/{judge_model}/iter5/{essay_model}/` |

주의: 현재 샘플 데이터는 `data/selected_prompt_jsons_100/`에 있고, 일부 에세이 judge 스크립트 기본값은 `data/selected_prompt_jsons/`이다. 실행 전 둘 중 하나로 통일해야 한다.

### SummEval

| 용도 | 경로 |
|---|---|
| judge 입력 | `summeval_judge_input/{system_name}/` |
| single judge 결과 | `summeval_judge_results/single_judge/{judge_model}/{system_name}/` |
| mad1 결과 | `summeval_judge_results/mad1/{judge_model}/{system_name}/` |
| mad2 결과 | `summeval_judge_results/mad2_iter/{judge_model}/iter5/{system_name}/` |
| RQ1 집계 | `stats/summeval_metrics_all.json` |

주의: repo에는 과거 명칭인 `summeval_judge_results/mad_c/`, `summeval_judge_results/mad_a_iter/` 결과가 남아 있다. 현재 스크립트는 `mad1`, `mad2_iter`를 표준으로 사용한다.

## 구현 상태

| 구분 | 에세이 | SummEval |
|---|---|---|
| 데이터 준비 | `prepare_nikl_essay_dataset.py` 있음 | `prepare_summeval.py` 있음 |
| single | `legacy_rationale_single_judge.py`, `single_*` 추론 있음 | `single_judge_summeval.py` 있음 |
| mad1 | `mad1_critic_defender.py` 있음 | `mad1_critic_defender_summeval.py` 있음 |
| mad2 iter | `mad2_consensus_iter.py` 있음 | `mad2_consensus_iter_summeval.py` 있음 |
| RQ1 집계 | direct scoring 중심, 통합표 보강 필요 | `rq1_score_distribution_summeval.py` 있음 |
| RQ2 분석 | `rq2_mad2_oscillation.py` 있음 | 전용 스크립트 필요 |
| RQ3 text-only | `mad2_text_only.py`, `analysis/rq3_anchoring.py` 있음 | 전용 실험/분석 스크립트 필요 |
| gold 성능 | `essay_scoring_performance.py` 있음 | 전용 스크립트 필요 |

## 우선순위

### P0. 경로 정합성 고정

- [ ] 에세이 데이터 경로를 하나로 고정한다.
  - 권장: `data/selected_prompt_jsons_100/`를 표준으로 쓰고, 스크립트 기본값을 이 경로로 맞춘다.
  - 호환이 필요하면 `data/selected_prompt_jsons` 심볼릭 링크 또는 복사본을 만든다.
- [ ] SummEval 결과 경로를 현재 스크립트 기준으로 통일한다.
  - 표준: `mad1`, `mad2_iter`
  - 과거 결과 `mad_c`, `mad_a_iter`는 legacy 결과로만 취급한다.
- [ ] 실행 생성물은 git 추적 대상에서 제외한다.
  - `debug/raw_responses/`, `__pycache__/`, `*.pyc`, 오타 경로 `lagacy/`

### P1. 에세이 결과 산출

- [ ] `inference_results/{gemma,qwen,llama,gpt}/`를 최신 샘플 600개 기준으로 산출한다.
- [ ] single judge 결과를 산출한다.

```bash
python src/essay/legacy_rationale_single_judge.py --judge-model gpt
python src/essay/legacy_rationale_single_judge.py --judge-model gemma
python src/essay/legacy_rationale_single_judge.py --judge-model qwen
```

- [ ] mad1 결과를 산출한다.

```bash
python src/essay/mad1_critic_defender.py
```

- [ ] mad2 iter5 결과를 산출한다.

```bash
python src/essay/mad2_consensus_iter.py --judge-model gpt --iterations 5
python src/essay/mad2_consensus_iter.py --judge-model gemma --iterations 5
python src/essay/mad2_consensus_iter.py --judge-model qwen --iterations 5
```

### P2. SummEval 결과 보완

- [ ] SummEval 입력 1,600개를 재생성하거나 현재 입력 개수를 검증한다.

```bash
python src/summeval/prepare_summeval.py --sample-size 100 --seed 42
```

- [ ] single/mad1/mad2 iter5를 judge model별로 산출한다.

```bash
python src/summeval/single_judge_summeval.py --judge-model gpt --workers 8
python src/summeval/mad1_critic_defender_summeval.py --judge-model gpt --workers 8
python src/summeval/mad2_consensus_iter_summeval.py --judge-model gpt --iterations 5 --workers 8
```

동일 명령을 `gemma`, `qwen`에도 반복한다. 최종 실험 기준은 `iter5`이며, 미완료 judge model의 `iter5` 결과가 우선 보완 대상이다.

### P3. RQ1 통합표

- [ ] 에세이 RQ1 스크립트를 `single_judge`, `mad1`, `mad2_iter/iter5`를 읽는 형태로 확장한다.
- [ ] SummEval RQ1 집계는 현재 표준 경로(`mad1`, `mad2_iter`) 기준으로 실행한다.

```bash
python src/summeval/rq1_score_distribution_summeval.py --all
```

최종 표에는 최소한 `count`, `mean`, `std`, `range`, `iqr`, `mode_frequency`, `wasserstein_from_uniform`을 포함한다.

### P4. RQ2 분석

- [ ] 에세이 RQ2를 judge model별 `iter5` 기준으로 산출한다.

```bash
python src/essay/rq2_mad2_oscillation.py --judge-model gpt
python src/essay/rq2_mad2_oscillation.py --judge-model gemma
python src/essay/rq2_mad2_oscillation.py --judge-model qwen
```

- [ ] SummEval용 RQ2 스크립트를 추가한다.
  - 입력: `summeval_judge_results/mad2_iter/{judge_model}/iter5/`
  - 출력: `stats/rq2_summeval_oscillation_{judge_model}.json/.md`
  - 지표: `flip_count`, `delta_strict`, `delta_lenient`, `convergence_final`

### P5. RQ3 ablation

- [ ] 에세이 text-only 결과를 산출한다.

```bash
python src/essay/mad2_text_only.py --judge-model gpt --iterations 5
python src/essay/mad2_text_only.py --judge-model gemma --iterations 5
python src/essay/mad2_text_only.py --judge-model qwen --iterations 5
```

- [ ] `mad2_text_only.py`에서 공유 텍스트 내부 숫자 표현까지 마스킹한다.
- [ ] `analysis/rq3_anchoring.py`의 기본 에세이 모델 목록에 `qwen`을 포함한다.
- [ ] SummEval text-only 실험 스크립트와 full vs text-only 분석 스크립트를 추가한다.

### P6. 보조 실험과 gold 성능

- [ ] mad1 defender bias 분석 스크립트를 추가한다.
  - `defender_win_rate`
  - `critic_major_issue_count`와 `final_score` 상관
  - `defender_rebuttal_success_rate`와 `score_change` 상관
- [ ] 에세이 human average 대비 성능표를 최신 결과로 산출한다.

```bash
python src/essay/essay_scoring_performance.py --input-dir inference_results/gemma
```

- [ ] SummEval human score 대비 RMSE/MAE/Pearson/Spearman 평가 스크립트를 추가한다.

## 논문 산출물 체크리스트

- [ ] Introduction 초안
- [ ] Background & Related Work 초안
- [ ] Method 섹션에 최종 실험 경로/모델/샘플 수 반영
- [ ] RQ1 표와 해석 문단
- [ ] RQ2 표와 해석 문단
- [ ] RQ3 표와 해석 문단
- [ ] Defender bias 보조 실험 표
- [ ] Essay와 SummEval을 함께 보여주는 최종 비교표
- [ ] Discussion / Limitation / Conclusion

## 정리된 불필요 파일 정책

삭제 또는 ignore 대상:

- `__pycache__/`, `*.pyc`
- `debug/raw_responses/`
- 오타 경로의 중복 보관본 `lagacy/`
- 새 `plan.md`와 중복되는 과거 실행 메모

보존 대상:

- `src/essay/legacy/`: 현재 본문 실험 밖 코드이지만 재현성 확인용으로 보존
- `stats/*.json`, `stats/*.md`: 과거 결과라도 논문 초안 작성 중 비교 근거로 보존
- `summeval_judge_results/mad_c`, `summeval_judge_results/mad_a_iter`: 현재 표준 경로는 아니지만 기존 실행 결과로 보존
