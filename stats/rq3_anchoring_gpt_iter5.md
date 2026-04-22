# RQ3 Anchoring Analysis (gpt judge, iter5)

## 1. adjustment_notes 패턴

| n_notes | anchor_hits_total | logic_hits_total | anchor_hits_mean | logic_hits_mean | anchor_dominant_ratio | logic_dominant_ratio | anchor_only | logic_only | mixed | neither |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 21600 | 117924 | 149575 | 5.459 | 6.925 | 0.246 | 0.627 | 6 | 3 | 21591 | 0 |

## 2. Full vs Text-only

| condition | n | flip_count_mean | flip_rate | convergence_final_mean | initial_gap_mean | gap_reduction_mean |
|---|---:|---:|---:|---:|---:|---:|
| full | 2700 | 3.620 | 0.985 | 0.617 | 0.973 | 0.356 |
| text_only | 2700 | 3.874 | 0.990 | 0.928 | 0.973 | 0.045 |

## 3. Text-only - Full 차이

| flip_count_mean_diff | flip_rate_diff | convergence_final_mean_diff | gap_reduction_mean_diff |
|---:|---:|---:|---:|
| 0.253 | 0.005 | 0.311 | -0.311 |

## 4. Interpretation

adjustment_notes에서는 숫자/점수 관련 표현보다 내용 참조 표현이 더 많이 나타났다. 내용 우세 노트 비율이 앵커링 우세 노트 비율보다 높다. text-only에서도 flip이 줄지 않았고, 진동이 유지되거나 증가했다. 따라서 현재 데이터만 보면 숫자 앵커링 단독 가설은 강하게 지지되지 않으며, 텍스트 피드백 자체의 상호 모방/반복도 중요한 원인으로 보인다.
