# 주관적인 평가 주제에서의 MAD 기반 LLM Judge의 점수 분포 편향에 대하여

> **버전**: 0.3  
> **작성일**: 2026.04.23

---

## 논문 핵심 주장 (One-liner)

> *에세이 평가와 텍스트 요약 평가는 모두 정답이 존재하지 않고 인간 평가자 간 견해 차이가 구조적으로 발생하는 주관적 질적 평가 태스크라는 공통 속성을 지닌다. 이러한 주관적 평가 태스크에서 합의 기반 다중 에이전트 토론(mad2)을 LLM Judge로 사용할 경우, 점수 분포가 축소되며 이는 논리적 합의가 아닌 숫자 앵커링에 의한 진동 현상에서 비롯된다. 따라서 주관적 질적 평가 태스크에서 합의 기반 MAD는 평가의 변별력을 저하시키며, 그 적용을 지양해야 한다.*

---

## 인과 구조

```
[현상] 합의 mad2 적용 시 Judge 점수 분포 범위가 축소된다
            ↓ Why?
[원인] 합의 과정에서 에이전트 간 점수 역전 혹은 수렴 현상이 발생한다
            ↓ How?
[메커니즘] 에이전트가 상대방의 논리가 아닌 점수 숫자에 반응 (앵커링)하여
           점수가 중간값으로 수렴한다
            ↓ So what?
[함의] 주관적 Judge task에서 합의 기반 mad2는 변별력을 희생하는
       trade-off가 존재하며, 주관적 주제에 대한 평가에는 적합하지 않을 수 있다
```

---

## Research Questions

| RQ | 질문 | 대응 실험 |
|----|------|---------|
| **RQ1** | LLM Judge 방식(single / mad1 / mad2)에 따라 주관적 질적 평가 태스크(에세이 평가, 텍스트 요약 평가)의 점수 분포가 어떻게 달라지는가? | 실험 1 |
| **RQ2** | mad2에서 관찰되는 점수 분포 축소의 과정에서 두 에이전트간 점수의 역전 혹은 중간값으로 수렴하는 현상이 나타나는가? | 실험 2 |
| **RQ3** | 진동 현상의 원인은 논리적 설득인가, 숫자 앵커링인가? | 실험 3 |

---

## 논문 구조

### 1. Introduction

LLM Judge의 개념 설명
Multi-agent debate의 개념 설명
다중평가자의 고전적인 anchoring현상 서술
주관적인 평가 주제에서의 anchoring현상이 왜 문제가 되는지 서술

**연구 기여**:
1. 주관적 질적 평가 태스크(에세이, 텍스트 요약)에서 single / mad1 / mad2 방식의 점수 분포 체계적 비교
2. mad2에서 점수 역전 및 수렴 현상(진동) 규명
3. 진동의 원인으로 숫자 앵커링 효과 제시 (ablation 포함)
4. 주관적 평가에서 합의 기반 MAD 사용 시의 한계와 blind review 설계 필요성 제언

---

### 2. Background & Related Work
keyword: LLM Judge, Multi-Agent Debate, score anchoring

#### 2.1 LLM-as-a-Judge
- 단일 LLM을 평가자로 사용하는 연구 흐름
- 주요 한계: 위치 편향, 자기 선호 편향, 관대화 경향

#### 2.2 Multi-Agent Debate (MAD)
- Du et al. (2023) 등 MAD의 정확도 향상 효과
- 기존 MAD 적용 맥락: 사실 기반 QA, 추론 task
- **Gap**: 주관적 평가 task에서의 MAD 적용 연구 부재

#### 2.3 Score Anchoring
- Kahneman & Tversky의 고전적 앵커링 편향: 초기 수치 정보가 이후 판단에 영향
- 다중 평가자 환경에서의 앵커링: 선행 평가자의 점수가 후속 평가자 점수를 당기는 현상
- LLM 평가에서의 앵커링 관련 선행 연구 (있다면 인용)
- **Gap**: LLM 에이전트 간 상호작용에서 점수 숫자가 앵커로 작용하는지에 대한 연구 부재

---

### 3. Method

#### 3.1 Task 정의

본 논문은 주관적인 평가 주제에서의 MAD 방식의 LLM Judge의 현상을 관찰하기 위하여 2가지의 데이터셋을 사용함.

1. SummEval
- human score as gold-label
- composed of 100 articles with 16 mechanical summaries (total of 1,600)
- relevance, coherence, fluency, consistency로 평가항목 구성

2. 국어원 에세이 평가 말뭉치
- 평가자 평균(average)을 gold-label로 선택 (두 평가자 간 편차를 완충하기 위함)
- 에세이 주제 6개에 총 약 2,600개 에세이
- content, expression, organization으로 평가항목 구성

각 데이터셋에 대한 LLM Judge의 task는 다음과 같이 정의됨:
1. 요약문의 품질을 평가 기준에 따라 평가
2. 에세이를 주제 및 평가 기준에 따라 직접 평가

#### 3.2 실험 종류

모든 실험은 에세이 데이터셋과 SummEval 데이터셋에 대해 동일하게 수행됨.

1. single, mad1, mad2(iter=5)의 baseline 실험 (RQ1, RQ2)
2. mad2에서 iteration별 점수 관련 정보를 제거한 text-only 조건과 baseline 비교 (RQ3)

#### 3.3 Judge 방식 설계

**방식 1: Single Judge (Baseline)**
```
[평가 대상] → [Single LLM Judge] → 각 항목 점수 산출
```
- 변별력 기준선 확보 목적

**방식 2: 비판 중점 MAD (mad1)**
```
[평가 대상] → [Critic]    ─┐
                           ├→ [Final Judge] → 점수
             → [Defender] ─┘
```
- **Critic**: 평가 대상의 문제점(hallucination, generic 표현 등) 탐지
- **Defender**: Critic의 지적을 텍스트 근거로 반박
- **Final Judge**: 양측 논거를 종합하여 최종 점수 산출
- `winner_side` 기록: `critic` / `defender` / `tie`

**방식 3: 합의 중점 MAD (mad2)**
```
[평가 대상] → [Strict Judge]  ─┐ 독립 평가 (round_0)
             → [Lenient Judge] ─┘
                    ↓
             상호 참조 → 점수 수정 (round_1, 2, ...)
                    ↓
             평균값 집계 → 최종 점수
```
- **Strict Judge**: 모든 항목에 결함이 있다고 전제하고 평가
- **Lenient Judge**: 긍정적 가능성을 최대한 인정하며 평가
- Iteration: `iter=5`로 통일하여 실험

#### 3.4 데이터셋

##### 데이터셋 1: 한국어 에세이 평가 (주 도메인)

- **출처**: 국립국어원 2024년 글쓰기 채점 말뭉치
- **규모**: Q4~Q9 에세이 주제별 100개 × 6 = **600개**
- **Judge LLM**: Qwen3-9B, Gemma3-4B, GPT-4o-mini (3개 모델)
- **인간 평가**: 평가자 평균 (5점 척도) → gold-label
- **Judge 평가 기준**: `content` / `organization` / `expression`

##### 데이터셋 2: 영어 텍스트 요약 품질 평가 (도메인 일반화 검증)

- **출처**: SummEval (Fabbri et al., 2021)
- **규모**: CNN/DailyMail 뉴스 기사 + 16개 요약 시스템 생성 요약 → **100개 기사 서브샘플**
- **Judge LLM**: Qwen3-9B, Gemma3-4B, GPT-4o-mini (3개 모델)
- **Judge 평가 기준**: `coherence` / `consistency` / `fluency` / `relevance`
- **인간 평가**: 크라우드소싱 어노테이터 점수 (1~5) → gold-label
- **역할**: 에세이 실험과 동일한 MAD 방식 적용 → 앵커링 현상이 다른 주관적 도메인에서도 재현되는지 검증

##### 데이터셋 선택 근거

| 항목 | 에세이 평가 | SummEval 요약 |
|------|-----------|------------|
| 주관성 | 높음 (정답 없음) | 높음 (정답 없음) |
| Judge 기준 수 | 3개 | 4개 |
| 인간 gold label | 있음 | 있음 |
| 언어 | 한국어 | 영어 |
| 모델 간 성능 차이 유지 | 있음 | 있음 |

두 데이터셋 모두 평가 기준에 따른 Judge 모델 간 성능 차이가 유지됨이 사전 확인되었으므로, Judge 방식 간 변별력 비교의 신뢰성 확보가 가능함.

---

### 4. 실험 설계

#### 실험 1: Judge 방식별 점수 분포 비교 (RQ1)

**목적**: single, mad1, mad2(iter=5) 간 점수 분포 차이 정량화

**측정 지표**:

| 지표 | 수식 | 의미 |
|------|------|------|
| Range | `max(scores) - min(scores)` | 분포 범위 |
| Std Dev | `σ(scores)` | 분산 |
| IQR | `Q3 - Q1` | 사분위 범위 |

**기대 결과**: mad2에서 Range와 Std Dev가 single 대비 유의미하게 축소됨

**적용 데이터셋**: 에세이 및 SummEval에 동일하게 적용

---

#### 실험 2: 합의 MAD 진동 현상 분석 (RQ2)

**목적**: mad2의 round 간 점수 변화 패턴을 정량화하여 진동 현상 규명

**측정 지표**:

```python
delta_strict_r  = |s_r - s_{r-1}|    # strict 에이전트 round 간 변화량
delta_lenient_r = |l_r - l_{r-1}|    # lenient 에이전트 round 간 변화량

# 진동 횟수: strict > lenient 관계가 역전된 횟수
flip_count = Σ 1[sign(s_r - l_r) ≠ sign(s_{r-1} - l_{r-1})]

# 최종 수렴 정도
convergence_final = |s_final - l_final|
```

**분석 방법**:
- 전체 샘플의 `flip_count` 분포 → 진동이 일반적 현상인지 검증
- `delta` 방향 분석: strict는 주로 올라가는가, lenient는 주로 내려가는가?
  - 대칭 수렴이면 → 실제 합의
  - 비대칭 수렴이면 → 한 에이전트가 지배적

**기대 결과**: flip_count > 0인 샘플 비율이 높고, delta 방향이 비대칭

**적용 데이터셋**: 에세이 및 SummEval에 동일하게 적용

---

#### 실험 3: 진동 원인 분석 — 앵커링 vs 논리적 설득 (RQ3)

**목적**: 에이전트가 상대방의 *점수 숫자*에 반응하는지, *근거 내용*에 반응하는지 구분

**방법 A: adjustment_notes 텍스트 분석 (현재 데이터 활용 가능)**

```python
anchor_pattern  = ["점수", "X점", "높다", "낮다", "조정"]   # 숫자 반응
logic_pattern   = ["근거", "rationale", "텍스트", "언급", "구체적"]  # 내용 반응

anchor_ratio = anchor_mentions / total_mentions
```

**방법 B: Ablation 실험 (text-only 조건)**

| 조건 | 공유 정보 | 목적 |
|------|---------|------|
| Full | 점수 + 근거 텍스트 공유 | 현재 설계 (baseline) |
| Text-only | 근거 텍스트만 공유, 점수 숨김 | 앵커링 효과 차단 |

- 두 조건의 `flip_count`와 `convergence_final` 비교
- Text-only 조건에서 flip이 감소하면 → **앵커링 효과 확인** (모델 근거 내 숫자 관련 표현도 마스킹)

**기대 결과**: 숫자 참조 비율 > 내용 참조 비율, Text-only 조건에서 flip 감소

**적용 데이터셋**: 에세이 및 SummEval에 동일하게 적용

---

#### (보조) 실험 4: 비판 MAD Defender 편향 분석

**목적**: mad1에서 Defender가 압도적으로 승리하는 원인 분석

**측정 지표**:
```python
defender_win_rate = Defender 승리 횟수 / 전체 판정 횟수

corr(critic_major_issue_count, final_score)
corr(defender_rebuttal_success_rate, score_change)
```

**기대 결과**: defender_win_rate가 유의미하게 0.5 초과, Critic major_issue와 최종 점수 간 약한 상관관계

---

### 5. Discussion

#### 5.1 주관적 Judge task에서의 MAD trade-off

```
single : 변별력 높음 / 단일 관점 편향 위험
mad1   : Defender 편향으로 관대화 경향
mad2   : 앵커링 기반 진동 → 중간값 수렴 → 변별력 저하
```

→ 주관적 평가에서 MAD는 *정확도 향상*보다 *점수 평준화*를 유도하는 경향이 있음

#### 5.2 앵커링 효과의 이론적 함의

- Kahneman의 앵커링 편향이 LLM 에이전트 간 상호작용에서도 나타남
- mad2 설계 시 **점수 숨김(blind review)** 방식이 필요함을 시사

#### 5.3 한계

- **Judge 모델 범위**: 본 실험에서 사용된 Judge 모델은 Qwen3-9B, Gemma3-4B, GPT-4o-mini 3종에 한정되며, 다른 모델 군(예: LLaMA 계열, Claude 등)에서 동일한 앵커링 현상이 재현되는지는 미검증
- **Gold-label 부재에 따른 절대 정확성 검증 불가**: 에세이 평가의 특성상 절대적 정답이 존재하지 않으므로, Judge 점수의 정확성은 인간 평가자 평균과의 상관을 통한 상대 비교로만 주장 범위를 한정함
- **mad2 역할 설정의 영향**: Strict/Lenient라는 역할 설정 자체가 앵커링 현상을 구조적으로 유도하는 요인일 수 있음. 역할 설정 방식이 달라질 경우 결과가 상이할 가능성을 배제할 수 없음

---

### 6. Conclusion

에세이 평가와 텍스트 요약 평가는 정답이 존재하지 않고 인간 평가자 간 견해 차이가 구조적으로 발생한다는 점에서 공통적으로 주관적 질적 평가 태스크에 해당한다. 본 논문은 이 공통 속성을 가진 두 도메인에서 Judge 방식(single, mad1, mad2)별 점수 분포와 변별력을 체계적으로 비교하였다.

실험 결과, 두 도메인 모두에서 mad2는 single 대비 점수 분포를 일관되게 축소시켰으며, 이는 에이전트 간 논리적 합의가 아닌 점수 숫자에 대한 앵커링에 의한 진동 현상에서 비롯됨을 확인하였다. 이는 주관적 질적 평가 태스크에서 합의 기반 MAD가 정확도 향상보다 점수 평준화를 유도한다는 것을 시사하며, 이러한 설계를 사용할 경우 점수 숨김(blind review) 방식의 도입이 필요함을 제언한다.
