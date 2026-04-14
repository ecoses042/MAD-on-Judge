# 에세이 채점 근거 평가에서 단일 및 토론 기반 LLM Judge의 판단 분포와 변별력 분석: 비판과 합의 관점에서

> **버전**: 0.1 (초안)  
> **작성일**: 2026.04.13

---

## 논문 핵심 주장 (One-liner)

> *합의 기반 다중 에이전트 토론(MAD)을 LLM Judge로 사용할 경우, 채점 근거 평가 점수의 분포가 축소되며 이는 논리적 합의가 아닌 숫자 앵커링에 의한 진동 현상에서 비롯된다.*

---

## 인과 구조

```
[현상] 합의 MAD 적용 시 Judge 점수 분포 범위가 축소된다
            ↓ Why?
[원인] 합의 과정에서 에이전트 간 점수 진동(oscillation)이 발생한다
            ↓ How?
[메커니즘] 에이전트가 상대방의 논리가 아닌 점수 숫자에 반응 (앵커링)하여
           점수가 중간값으로 수렴한다
            ↓ So what?
[함의] 주관적 Judge task에서 합의 기반 MAD는 변별력을 희생하는
       trade-off가 존재하며, 에세이 채점 근거 평가에 적합하지 않을 수 있다
```

---

## Research Questions

| RQ | 질문 | 대응 실험 |
|----|------|---------|
| **RQ1** | LLM Judge 방식(Single / 비판 MAD / 합의 MAD)에 따라 채점 근거 평가의 점수 분포가 어떻게 달라지는가? | 실험 1 |
| **RQ2** | 합의 MAD에서 관찰되는 점수 분포 축소의 원인은 무엇인가? 진동 현상이 존재하는가? | 실험 2 |
| **RQ3** | 진동 현상의 원인은 논리적 설득인가, 숫자 앵커링인가? | 실험 3 |
| **RQ4** | Iteration 수 증가는 진동을 완화하는가, 심화하는가? | 실험 4 |

---

## 논문 구조

### 1. Introduction

- **배경**: 에세이 자동 채점에서 *점수*만큼 *채점 근거의 품질*이 중요하다
- **문제 제기**: 채점 근거처럼 정답이 없는 주관적 결과물을 Judge로 평가할 때, 어떤 방식이 적합한가?
- **기존 연구의 한계**: LLM Judge는 주로 단일 모델로 사용되거나, MAD를 최종 답변 생성에 적용함. 그러나 **채점 근거 평가라는 주관적 메타 평가 맥락**에서 Judge 방식이 어떤 영향을 미치는지는 연구 부족
- **연구 기여**:
  1. 에세이 채점 근거 평가에서 3가지 Judge 방식의 점수 분포 비교
  2. 합의 MAD에서 점수 진동 현상 규명
  3. 진동의 원인으로 숫자 앵커링 효과 제시
  4. 비판 MAD의 Defender 편향 문제 분석

---

### 2. Background & Related Work

#### 2.1 LLM-as-a-Judge
- 단일 LLM을 평가자로 사용하는 연구 흐름
- 주요 한계: 위치 편향, 자기 선호 편향, 관대화 경향

#### 2.2 Multi-Agent Debate (MAD)
- Du et al. (2023) 등 MAD의 정확도 향상 효과
- 기존 MAD 적용 맥락: 사실 기반 QA, 추론 task
- **Gap**: 주관적 평가 task에서의 MAD 적용 연구 부재

#### 2.3 에세이 자동 채점 (Automated Essay Scoring)
- 기존 AES 연구: 점수 예측 중심 (RMSE, Spearman's ρ)
- LLM 기반 채점 근거 생성 연구
- **Gap**: 생성된 채점 근거의 *품질 평가 방법론* 연구 부재

---

### 3. Method

#### 3.1 Task 정의

```
에세이 → [채점 LLM] → (점수 + 근거) → [LLM Judge] → 채점 품질 점수
```

- **채점 LLM**: 에세이를 입력받아 Content / Organization / Expression 영역별 점수(1~5)와 근거 생성
- **LLM Judge**: 채점 결과물의 품질을 4가지 기준으로 평가

#### 3.2 Judge 평가 기준 (4가지)

| 항목 | 정의 |
|------|------|
| `domain_match` | 근거가 해당 평가 영역의 기준을 사용하는가 |
| `score_rationale_consistency` | 점수와 근거 설명이 논리적으로 일치하는가 |
| `specificity` | 근거가 에세이의 구체적 문장/표현을 지칭하는가 |
| `groundedness` | 근거가 실제 에세이 텍스트에 기반하는가 |

#### 3.3 Judge 방식 설계

**방식 1: Single Judge (Baseline)**
```
[채점 결과] → [Single LLM Judge] → 4개 항목 점수 산출
```
- 비판적 성향 강조 프롬프트 적용
- 변별력 기준선 확보 목적

**방식 2: 비판 중점 MAD (MAD-C)**
```
[채점 결과] → [Critic]    ─┐
                           ├→ [Final Judge] → 점수
             → [Defender] ─┘
```
- **Critic**: 근거의 문제점(hallucination, generic 표현 등) 탐지
- **Defender**: Critic의 지적을 rationale 텍스트 근거로 반박
- **Final Judge**: 양측 논거를 종합하여 최종 점수 산출
- `winner_side` 기록: `critic` / `defender` / `tie`

**방식 3: 합의 중점 MAD (MAD-A)**
```
[채점 결과] → [Strict Judge]  ─┐ 독립 평가 (round_0)
             → [Lenient Judge] ─┘
                    ↓
             상호 참조 → 점수 수정 (round_1, 2, ...)
                    ↓
             평균값 집계 → 최종 점수
```
- **Strict Judge**: 모든 근거에 결함이 있다고 전제하고 평가
- **Lenient Judge**: 긍정적 가능성을 최대한 인정하며 평가
- Iteration: `iter=3`, `iter=5` 두 조건으로 실험

#### 3.4 데이터셋

- **출처**: 국립국어원 2024년 글쓰기 채점 말뭉치
- **규모**: Q4~Q9 에세이 주제별 50개 × 6 = **300개**
- **채점 LLM**: GPT-4o-mini, Qwen3-9B, Gemma3-4B, LLaMA3-8B (4개 모델)
- **Judge LLM**: GPT-4o-mini (+ 추가 모델 1개 권장)
- **인간 평가**: 평가자 1, 평가자 2, 평균 (5점 척도)

---

### 4. 실험 설계

#### 실험 1: Judge 방식별 점수 분포 비교 (RQ1)

**목적**: Single Judge, MAD-C, MAD-A(iter=3), MAD-A(iter=5) 간 점수 분포 차이 정량화

**측정 지표**:

| 지표 | 수식 | 의미 |
|------|------|------|
| Range | `max(scores) - min(scores)` | 분포 범위 |
| Std Dev | `σ(scores)` | 분산 |
| IQR | `Q3 - Q1` | 사분위 범위 |
| 변별력 | 모델 간 Spearman's ρ 순위 보존율 | 모델 순위 구분 능력 |

**기대 결과**: 합의 MAD에서 Range와 Std Dev가 Single Judge 대비 유의미하게 축소됨

**시각화**: 방식별 violin plot (4개 Judge 조건 × 4개 Judge 항목)

---

#### 실험 2: 합의 MAD 진동 현상 분석 (RQ2)

**목적**: 합의 MAD의 round 간 점수 변화 패턴을 정량화하여 진동 현상 규명

**측정 지표**:

```python
# 에세이 i, round r에서의 strict/lenient 점수를 s_r, l_r로 표기

delta_strict_r  = |s_r - s_{r-1}|    # strict 에이전트 round 간 변화량
delta_lenient_r = |l_r - l_{r-1}|    # lenient 에이전트 round 간 변화량

# 진동 횟수: strict > lenient 관계가 역전된 횟수
flip_count = Σ 1[sign(s_r - l_r) ≠ sign(s_{r-1} - l_{r-1})]

# 최종 수렴 정도
convergence_final = |s_final - l_final|
```

**분석 방법**:
- 300개 에세이 전체의 `flip_count` 분포 → 진동이 일반적 현상인지 검증
- `delta` 방향 분석: strict는 주로 올라가는가, lenient는 주로 내려가는가?
  - 대칭 수렴이면 → 실제 합의
  - 비대칭 수렴이면 → 한 에이전트가 지배적

**기대 결과**: flip_count > 0인 에세이 비율이 높고, delta 방향이 비대칭

---

#### 실험 3: 진동 원인 분석 — 앵커링 vs 논리적 설득 (RQ3)

**목적**: 에이전트가 상대방의 *점수 숫자*에 반응하는지, *근거 내용*에 반응하는지 구분

**방법 A: adjustment_notes 텍스트 분석 (현재 데이터 활용 가능)**

```python
# adjustment_notes에서 다음 두 패턴의 빈도 측정
anchor_pattern  = ["점수", "X점", "높다", "낮다", "조정"]   # 숫자 반응
logic_pattern   = ["근거", "rationale", "텍스트", "언급", "구체적"]  # 내용 반응

anchor_ratio = anchor_mentions / total_mentions  # 앵커링 비율
```

**방법 B: Ablation 실험 (추가 실험)**

| 조건 | 공유 정보 | 목적 |
|------|---------|------|
| Full | 점수 + 근거 텍스트 공유 | 현재 설계 (baseline) |
| Text-only | 근거 텍스트만 공유, 점수 숨김 | 앵커링 효과 차단 |

- 두 조건의 `flip_count`와 `convergence_final` 비교
- Text-only 조건에서 flip이 감소하면 → **앵커링 효과 확인**

**기대 결과**: adjustment_notes에서 숫자 참조 비율 > 내용 참조 비율, Text-only 조건에서 flip 감소

---

#### 실험 4: Iteration 수에 따른 수렴 패턴 (RQ4)

**목적**: iteration=3 vs iter=5에서 진동 및 수렴 정도 비교

**측정 지표**:

```python
# 수렴 속도: 처음으로 |s_r - l_r| < threshold(=0.5)가 되는 round
stabilization_round = min{r : |s_r - l_r| < 0.5}

# iteration별 최종 분포 비교
range_iter3 vs range_iter5
flip_count_iter3 vs flip_count_iter5
```

**기대 결과**: iter=5가 더 좁은 분포를 만들지만, flip_count는 오히려 증가할 수 있음 (진동이 더 많이 일어난 후 수렴)

---

#### (보조) 실험 5: 비판 MAD Defender 편향 분석

**목적**: 비판 MAD에서 Defender가 압도적으로 승리하는 원인 분석

**측정 지표**:
```python
defender_win_rate = Defender 승리 횟수 / 전체 판정 횟수

# Critic major_issue 수와 최종 점수 상관관계
corr(critic_major_issue_count, final_score)

# Defender rebuttal 성공률과 점수 변화
corr(defender_rebuttal_success_rate, score_change)
```

**기대 결과**: defender_win_rate가 유의미하게 0.5 초과, Critic major_issue와 최종 점수 간 약한 상관관계

---

### 5. Discussion

#### 5.1 주관적 Judge task에서의 MAD trade-off

```
Single Judge : 변별력 높음 / 단일 관점 편향 위험
비판 MAD     : Defender 편향으로 관대화 경향
합의 MAD     : 앵커링 기반 진동 → 중간값 수렴 → 변별력 저하
```

→ 주관적 채점 근거 평가에서 MAD는 *정확도 향상*보다 *점수 평준화*를 유도하는 경향이 있음

#### 5.2 앵커링 효과의 이론적 함의

- Kahneman의 앵커링 편향이 LLM 에이전트 간 상호작용에서도 나타남
- 합의 MAD 설계 시 **점수 숨김(blind review)** 방식이 필요함을 시사

#### 5.3 한계

- Gold label 부재: Judge 점수의 절대적 정확성 검증 불가 → 방식 간 *상대적 비교*로 주장 범위 한정
- 단일 Judge 모델(GPT-4o-mini) 의존: 모델별 일반화 한계
- 에세이 주제별 편차 미통제: prompt_id별 층화 분석 필요

---

### 6. Conclusion

- 에세이 채점 근거 평가에서 Judge 방식별 점수 분포와 변별력을 체계적으로 비교
- 합의 MAD의 점수 축소 현상의 원인으로 **진동 현상과 앵커링 효과** 규명
- 주관적 Judge task에서 blind review 기반 MAD 설계의 필요성 제시

---

## 실험 일정 및 우선순위

| 우선순위 | 실험 | 필요 데이터 | 예상 비용 |
|---------|------|-----------|---------|
| ★★★ | 실험 1 (분포 비교) | 300개 × 4 모델 × 4 방식 | 높음 |
| ★★★ | 실험 2 (진동 분석) | 실험 1의 합의 MAD 출력 재활용 | 낮음 |
| ★★☆ | 실험 3-A (텍스트 분석) | 실험 1의 adjustment_notes 재활용 | 없음 |
| ★★☆ | 실험 4 (iteration 비교) | iter=5 추가 실험 필요 | 중간 |
| ★☆☆ | 실험 3-B (ablation) | Text-only 조건 추가 실험 | 중간 |
| ★☆☆ | 실험 5 (Defender 편향) | 실험 1의 비판 MAD 출력 재활용 | 낮음 |

> **핵심 전략**: 실험 1 데이터를 최대한 재활용하여 실험 2, 3-A, 5는 추가 API 비용 없이 진행 가능

---

## 측정 지표 요약

| 지표 | 관련 RQ | 측정 대상 |
|------|---------|---------|
| Range, Std, IQR | RQ1 | Judge 점수 분포 |
| Spearman's ρ 순위 보존 | RQ1 | 모델 간 변별력 |
| flip_count | RQ2 | 진동 횟수 |
| delta (strict/lenient) | RQ2 | round 간 변화량 |
| convergence_final | RQ2, RQ4 | 최종 수렴 정도 |
| anchor_ratio | RQ3 | 앵커링 vs 논리 반응 비율 |
| stabilization_round | RQ4 | 수렴 속도 |
| defender_win_rate | 보조 | Defender 편향 정도 |