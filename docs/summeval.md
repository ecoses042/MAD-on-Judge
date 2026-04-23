# SummEval 파이프라인 설명

`src/summeval/` 에 구현된 코드의 전체 흐름, 사용 데이터셋, judge 방식을 설명한다.

---

## 목적

한국어 에세이 judge 파이프라인(`src/essay/`)에서 관찰한 **MAD 진동·앵커링 현상**이  
영어 요약 품질 평가 도메인에서도 재현되는지 확인한다 (도메인 일반화 검증).

에세이 파이프라인과 1:1 대응하는 세 가지 judge 방식(Single / MAD-C / mad2 iter)을 동일하게 실행하고,  
결과 메트릭을 비교한다.

---

## 데이터셋: `mteb/summeval`

HuggingFace Hub의 `mteb/summeval` (test split)을 사용한다.

| 항목 | 내용 |
|------|------|
| 출처 | [SummEval: Re-evaluating Summarization Evaluation (Fabbri et al., 2021)](https://arxiv.org/abs/2007.12626) |
| 원문 | CNN/DailyMail 뉴스 기사 100편 |
| 요약 시스템 | 기사 1편당 16개 요약 시스템(machine_summaries) |
| 인간 평가 | 각 (기사, 시스템) 쌍에 대해 3명이 4개 기준으로 1~5점 평가 |
| 평가 기준 | coherence / consistency / fluency / relevance |
| 총 데이터 포인트 | 100기사 × 16시스템 = 1,600 (샘플 크기 변경 가능) |

### 4개 평가 기준 정의

| 기준 | 의미 |
|------|------|
| **coherence** | 요약이 잘 구조화되고 정보 흐름이 논리적인가 |
| **consistency** | 원문에 없는 사실을 포함하지 않는가 (hallucination 여부) |
| **fluency** | 문법 오류 없이 자연스럽게 읽히는가 |
| **relevance** | 원문의 핵심 내용을 빠짐없이, 불필요한 세부사항 없이 담았는가 |

### gold 레이블 계산

`prepare_summeval.py`가 3명 인간 평가자 점수를 평균내어 `gold` 필드에 저장한다.

```json
"gold": {
  "coherence": 4.33,
  "consistency": 5.0,
  "fluency": 4.67,
  "relevance": 3.67
}
```

---

## 전체 데이터 흐름

```
HuggingFace mteb/summeval
        │
        ▼  prepare_summeval.py
summeval_judge_input/
  {system_name}/
    {article_id}.json          ← article_text, summary_text, gold 포함
        │
        ├──▶ single_judge_summeval.py
        │         └──▶ summeval_judge_results/single_judge/{judge_model}/{system_name}/{article_id}.json
        │
        ├──▶ mad1_critic_defender_summeval.py
        │         └──▶ summeval_judge_results/mad1/{judge_model}/{system_name}/{article_id}.json
        │
        └──▶ mad2_consensus_iter_summeval.py
                  └──▶ summeval_judge_results/mad2_iter/{judge_model}/iter{N}/{system_name}/{article_id}.json
                            │
                            ▼  rq1_score_distribution_summeval.py
                    stats/summeval_metrics_all.json
```

---

## 스크립트별 로직 흐름

### 0. `prepare_summeval.py` — 입력 데이터 생성

```
load_dataset("mteb/summeval", split="test")
        │
        ▼ shuffle(seed) → select(sample_size)
        │
        ▼ 기사 1편 × 16시스템 → 16개 JSON 파일 생성
        │
        ▼ summeval_judge_input/{system_name}/{article_id}.json
```

출력 JSON 구조:
```json
{
  "article_id": "...",
  "system_name": "M0",
  "article_text": "뉴스 기사 원문",
  "summary_text": "해당 시스템의 요약문",
  "gold": { "coherence": 4.33, "consistency": 5.0, "fluency": 4.67, "relevance": 3.67 }
}
```

---

### 1. `single_judge_summeval.py` — Single Judge

단일 LLM 한 번의 호출로 4개 기준을 한꺼번에 평가한다. 에세이 파이프라인의 비교 기준선에 해당.

```
입력 파일 (article_text + summary_text)
        │
        ▼ SINGLE_JUDGE_SYSTEM 프롬프트
        │   "You are a strict evaluator. Score coherence/consistency/fluency/relevance."
        │
        ▼ LLM 호출 1회 (temperature=0.0)
        │
        ▼ JSON 파싱 + 점수 clamp(1~5)
        │
        ▼ compute_overall() — 4개 평균, 최솟값 기반 상한 적용
        │
        ▼ 출력 저장
```

출력 JSON:
```json
{
  "article_id": "...",
  "system_name": "M0",
  "gold": { ... },
  "judge_model": "gpt",
  "judge": {
    "coherence": 4,
    "consistency": 5,
    "fluency": 4,
    "relevance": 3,
    "overall_judge": 4.0
  }
}
```

병렬화: `--workers N`으로 ThreadPoolExecutor 사용. 파일 단위 체크포인트(출력 파일 존재 시 skip).

---

### 2. `mad1_critic_defender_summeval.py` — mad1 (Critic-Defender-Final Judge)

Critic → Defender → Final Judge 3단계 토론 구조.

```
입력 (article_text + summary_text)
        │
        ├──▶ [Critic] CRITIC_SYSTEM (temperature=0.1)
        │     "공격적 오류 탐지자. 약점 2개 이상 찾아라."
        │     출력: { major_issues, minor_issues, provisional_score }
        │
        ├──▶ [Defender] DEFENDER_SYSTEM (temperature=0.1)
        │     "공정한 방어자. 강점 1개 이상 제시."
        │     출력: { strengths, remaining_concerns, provisional_score }
        │
        └──▶ [Final Judge] FINAL_JUDGE_SYSTEM (temperature=0.0)
              입력: article_text + summary_text + critic 결과 + defender 결과
              "더 설득력 있는 쪽을 채택해 최종 점수 결정."
              출력: { coherence, consistency, fluency, relevance, overall_judge,
                      winner_side: {criterion: "critic|defender|tie"} }
```

Critic과 Defender는 **독립적으로 요약문을 평가**하고, Final Judge가 두 의견을 종합한다.  
`winner_side` 필드에 각 기준별로 어느 쪽 주장이 채택됐는지 기록된다.

출력 JSON:
```json
{
  "judge": {
    "critic":   { "major_issues": [...], "minor_issues": [...], "provisional_score": 2 },
    "defender": { "strengths": [...], "remaining_concerns": [...], "provisional_score": 4 },
    "final": {
      "coherence": 3, "consistency": 5, "fluency": 4, "relevance": 3,
      "overall_judge": 3.7,
      "winner_side": { "coherence": "critic", "consistency": "defender", ... }
    }
  }
}
```

---

### 3. `mad2_consensus_iter_summeval.py` — mad2 iter (반복 합의)

Strict judge와 Lenient judge가 서로의 평가를 보고 점수를 조정하는 과정을 N회 반복한다.

#### 전체 구조 (예: iterations=3, round_0~round_2)

```
round_0 (초기 평가, 병렬 실행, temperature=0.1)
  ├── Strict:  STRICT_JUDGE_SYSTEM  → { coherence, consistency, fluency, relevance, overall_judge, rationale_for_score }
  └── Lenient: LENIENT_JUDGE_SYSTEM → { coherence, consistency, fluency, relevance, overall_judge, rationale_for_score }

round_1 (상호 참조 조정, temperature=0.0)
  ├── Strict  ← Lenient의 round_0 결과 참조 → STRICT_JUDGE_ADJUST_SYSTEM
  └── Lenient ← Strict의 round_0 결과 참조  → LENIENT_JUDGE_ADJUST_SYSTEM
        각 출력: { coherence, ..., overall_judge, adjustment_notes }

round_2 (마지막 조정, 동일 구조)
  ├── Strict  ← Lenient의 round_1 결과 참조
  └── Lenient ← Strict의 round_1 결과 참조

final
  = round_{N-1}.strict 와 round_{N-1}.lenient 각 항목의 단순 평균
```

#### Strict vs Lenient 역할

| 역할 | 기본 태도 | 조정 원칙 |
|------|-----------|-----------|
| **Strict** | 결함을 전제하고 시작. hallucination → 즉시 1점. 구체적 근거 없으면 낮게. | 관대한 쪽 근거 중 요약문에 실제 있는 것만 인정해 점수 올릴 수 있음 |
| **Lenient** | 긍정적 해석 여지를 찾음. 방향만 맞으면 높게. 명백한 오류만 감점. | 엄격한 쪽 지적 중 실제 확인되는 결함은 인정해 점수 내릴 수 있음 |

#### FIRST_COMPLETED 병렬 처리 패턴

`run()` 내부에서 `ThreadPoolExecutor(max_workers=2)`와 `FIRST_COMPLETED` wait를 사용한다.  
한 side의 결과가 나오는 즉시 다음 라운드의 상대방 호출을 제출해 대기 시간을 최소화한다.

```
round_0 strict 완료 → 즉시 round_1 lenient 제출
round_0 lenient 완료 → 즉시 round_1 strict 제출
...
```

출력 JSON:
```json
{
  "judge": {
    "round_0": {
      "strict":  { "coherence": 2, "consistency": 4, "fluency": 4, "relevance": 2, "overall_judge": 3.0, "rationale_for_score": "..." },
      "lenient": { "coherence": 4, "consistency": 5, "fluency": 5, "relevance": 4, "overall_judge": 4.5, "rationale_for_score": "..." }
    },
    "round_1": {
      "strict":  { ..., "adjustment_notes": "..." },
      "lenient": { ..., "adjustment_notes": "..." }
    },
    "round_2": { ... },
    "final": { "coherence": 3.5, "consistency": 4.5, "fluency": 4.5, "relevance": 3.0, "overall_judge": 3.9 }
  }
}
```

---

### 4. `rq1_score_distribution_summeval.py` — 메트릭 집계

judge 결과 파일들을 읽어 분포 메트릭을 계산하고 `stats/summeval_metrics_all.json`에 저장한다.

집계 단위: **시스템(system_name)별** × **기준(coherence 등)별**

| `uses_final` | 읽는 필드 | 해당 실험 |
|---|---|---|
| `False` | `judge.coherence` 등 직접 | single_judge, mad1 |
| `True` | `judge.final.coherence` 등 | mad2_iter |

계산하는 메트릭:

| 메트릭 | 의미 |
|--------|------|
| `mean` | 평균 점수 |
| `std` | 표준편차 |
| `range` | max − min |
| `iqr` | Q3 − Q1 |
| `cv` | 변동계수 (std/mean) |
| `normalized_entropy` | 점수 분포 엔트로피 (1~5 기준, 균일분포=1.0) |
| `mode_freq` | 최빈값 비율 (높을수록 집중) |
| `wasserstein_from_uniform` | 균일분포와의 Wasserstein 거리 (scipy 설치 시) |

---

## `summeval_prompt_utils.py` 주요 함수

| 이름 | 역할 |
|------|------|
| `build_user_prompt(sample)` | `[article_text]\n...\n\n[summary_text]\n...` 형식으로 user 메시지 생성 |
| `build_adjust_prompt(sample, other_result, reviewer_label, instruction)` | user_prompt에 상대방 평가 결과 JSON을 덧붙여 조정용 프롬프트 생성 |
| `compute_overall(result)` | 4개 점수 평균. 최솟값이 1이면 상한 2.0, 2면 상한 3.0 적용 |

---

## 실행 순서

```bash
# 1. 데이터 준비 (1회)
python src/summeval/prepare_summeval.py --sample-size 100 --seed 42

# 2. Single Judge
python src/summeval/single_judge_summeval.py --judge-model gpt --workers 4

# 3. mad1
python src/summeval/mad1_critic_defender_summeval.py --judge-model gpt --workers 4

# 4. mad2 iter
python src/summeval/mad2_consensus_iter_summeval.py --judge-model gpt --iterations 3 --workers 4
python src/summeval/mad2_consensus_iter_summeval.py --judge-model gpt --iterations 5 --workers 4

# 5. 메트릭 집계
python src/summeval/rq1_score_distribution_summeval.py --all
# 또는 단일 실험
python src/summeval/rq1_score_distribution_summeval.py --exp single_judge/gpt
python src/summeval/rq1_score_distribution_summeval.py --exp mad2_iter/gpt/iter3
```

### judge_model 옵션

| 값 | 실제 모델 | 연결 방식 |
|----|-----------|-----------|
| `gpt` | gpt-4o-mini | OpenAI API (`.env`의 `OPENAI_API_KEY`) |
| `qwen` | qwen/qwen3.5-9b | LM Studio 로컬 (`http://localhost:1234/v1`) |
| `gemma` | google/gemma-3-4b | LM Studio 로컬 (`http://localhost:1234/v1`) |

---

## 에세이 파이프라인과의 비교

| 항목 | 에세이 (`src/essay/`) | SummEval (`src/summeval/`) |
|------|-----------------------|---------------------------|
| 입력 | essay_text + predicted_score + rationale | article_text + summary_text |
| 평가 단위 | domain(content/org/exp) × essay별 | 요약문 1개 → 4기준 한 번에 |
| 4개 기준 | domain_match / score_rationale_consistency / specificity / groundedness | coherence / consistency / fluency / relevance |
| 언어 | 한국어 | 영어 |
| gold 레이블 | 인간 채점자 5점 척도 | 인간 평가자 3명 평균 (1~5) |
| 병렬화 단위 | 에세이 파일 내 순차 | 파일 단위 ThreadPoolExecutor |
