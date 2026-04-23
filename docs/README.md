# Documentation Index

이 디렉토리는 현재 `outline.md` v0.3 기준 실험 파이프라인 사용법을 정리한다.

## Essay

| 문서 | 대상 |
|---|---|
| `prepare_nikl_essay_dataset_usage.md` | NIKL 원본에서 Q4~Q9, 주제별 100개 샘플 생성 |
| `legacy_rationale_single_judge_usage.md` | 에세이 single judge baseline |
| `mad2_consensus_iter_usage.md` | 에세이 mad2 iter3/iter5 full condition |
| `mad2_text_only_usage.md` | 에세이 mad2 text-only ablation |

## SummEval

| 문서 | 대상 |
|---|---|
| `summeval_pipeline_usage.md` | SummEval 준비, single, mad1, mad2, RQ1 집계 실행법 |
| `summeval.md` | SummEval 파이프라인 구조와 judge 로직 설명 |

## Current Naming

현재 표준 실험명은 다음과 같다.

| 논문 표기 | 코드/경로 표기 |
|---|---|
| single | `single_judge` |
| mad1 | `mad1_critic_defender`, `mad1` |
| mad2 | `mad2_consensus_iter`, `mad2_iter` |
| text-only ablation | `mad2_text_only` |

과거 명칭 `mad_c`, `mad_a_iter`는 legacy 결과 경로에만 남아 있으며, 새 실행과 새 집계는 `mad1`, `mad2_iter` 기준으로 맞춘다.
