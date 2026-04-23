# mad2_consensus_iter.py 사용 가이드

mad2의 Strict/Lenient 상호 조정 방식을 **N회 반복**으로 확장한 실험 스크립트.  
`--iterations N` 인자로 교환 횟수를 지정하며, 기본값은 3회다.

text-only ablation 버전은 [docs/mad2_text_only_usage.md](/mnt/c/Users/c/OneDrive/Desktop/NLP lab/제안서/말평/초기실험 코드/docs/mad2_text_only_usage.md:1)를 참고한다. 해당 변형도 각 라운드에서 strict/lenient 추론 2개를 병렬 실행한다.

---

## 실행 명령

```bash
python src/essay/mad2_consensus_iter.py [--model {gemma,qwen,llama,gpt}] [--judge-model {gpt,gemma,qwen}] [--iterations N]
```

### 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--model` | 없음 (전체 순차 실행) | 처리할 에세이 모델 |
| `--judge-model` | 없음 (gpt → gemma → qwen 순서대로 실행) | Judge로 사용할 모델 |
| `--iterations` | 없음 (3회 → 5회 순서대로 실행) | 총 교환 횟수 (최소 2) |

`--model`을 생략하면 기본적으로 모든 에세이 모델을 순차 처리한다. 단, `--judge-model qwen`과 함께 `--model`을 생략한 경우에는 `llama`, `gpt` 에세이 모델만 처리한다.

### judge-model 선택지

| 값 | 실제 모델 | 백엔드 |
|---|---|---|
| `gpt` | `gpt-4o-mini` | OpenAI API |
| `gemma` | `google/gemma-3-4b` | LM Studio (`http://localhost:1234/v1`) |
| `qwen` | `qwen/qwen3.5-9b` | LM Studio (`http://localhost:1234/v1`) |

> LM Studio 모델은 `response_format={"type": "json_object"}`를 지원하지 않으므로 해당 옵션을 사용하지 않는다. GPT 계열만 `response_format`을 사용한다.
> LM Studio 로컬 모델인 `gemma`, `qwen`은 `agent_sleep`, `domain_sleep`, 에세이 간 대기가 자동으로 `0.0`초로 설정된다.

---

## 실행 예시

```bash
# 모든 인자 생략: gpt→gemma→qwen judge × iter3→iter5 조합으로 전체 순차 실행
python src/essay/mad2_consensus_iter.py

# judge-model 생략, iterations 지정: gpt→gemma→qwen 순서로 3회씩 실행
python src/essay/mad2_consensus_iter.py --iterations 3

# GPT judge, gpt 에세이 모델만, 3회
python src/essay/mad2_consensus_iter.py --judge-model gpt --model gpt --iterations 3

# Gemma judge (LM Studio), 5회
python src/essay/mad2_consensus_iter.py --judge-model gemma --iterations 5

# Qwen judge (LM Studio), --model 생략 시 llama/gpt 에세이 모델만 처리, 5회
python src/essay/mad2_consensus_iter.py --judge-model qwen --iterations 5

# Qwen judge (LM Studio), llama 에세이 모델만, 3회
python src/essay/mad2_consensus_iter.py --judge-model qwen --model llama --iterations 3
```

---

## 입출력 구조

### 입력

| 경로 | 내용 |
|---|---|
| `inference_results/gemma/` | Gemma 채점 결과 |
| `inference_results/qwen/` | Qwen 채점 결과 |
| `inference_results/llama/` | LLaMA 채점 결과 |
| `inference_results/gpt/` | GPT 채점 결과 |
| `data/selected_prompt_jsons/` | 에세이 원본 (prompt_text, essay_text). 현재 샘플 생성 스크립트의 기본 출력은 `data/selected_prompt_jsons_100/`이므로 실행 전 경로를 통일해야 한다. |

### 출력

```
judge_results/mad2_iter/{judge_model}/iter{N}/{essay_model}/
```

예시 (`--judge-model gpt --iterations 3`):

```
judge_results/mad2_iter/
└── gpt/
    └── iter3/
        ├── gemma/
        ├── qwen/
        ├── llama/
        └── gpt/
```

출력 파일명은 입력 파일명과 동일하다.

---

## 라운드 구조

```
round_0: strict(독립 평가, temp=0.1)  ←→  lenient(독립 평가, temp=0.1)
round_1: strict(lenient_0 참조, temp=0.0)  ←→  lenient(strict_0 참조, temp=0.0)
...
round_N-1: strict(lenient_N-2 참조)  ←→  lenient(strict_N-2 참조)
final: round_N-1의 strict + lenient 평균
```

- **round_0**: 두 Judge가 서로 독립적으로 평가
- **round_1 이후**: 이전 라운드 상대방 결과를 참조하여 재조정
- **final**: 마지막 라운드의 strict/lenient 점수를 단순 평균

도메인당 API 호출 횟수 = `N × 2`회

---

## 출력 JSON 구조

```json
[
  {
    "essay_id": "GWGR2400103100.1",
    "status": "ok",
    "prediction": {
      "content":      { "score": 3.5, "rationale": "..." },
      "organization": { "score": 3.0, "rationale": "..." },
      "expression":   { "score": 4.0, "rationale": "..." }
    },
    "judge": {
      "content": {
        "round_0": {
          "strict":  { "domain_match": 3, "score_rationale_consistency": 3, "specificity": 3, "groundedness": 3, "overall_judge": 3.0 },
          "lenient": { "domain_match": 4, "score_rationale_consistency": 4, "specificity": 3, "groundedness": 4, "overall_judge": 3.8 }
        },
        "round_1": { "strict": { ... }, "lenient": { ... } },
        "final": {
          "domain_match": 3.5,
          "score_rationale_consistency": 3.5,
          "specificity": 3.0,
          "groundedness": 3.5,
          "overall_judge": 3.4
        }
      },
      "organization": { ... },
      "expression":   { ... }
    }
  }
]
```

---

## 평가 항목

각 도메인(`content`, `organization`, `expression`)에 대해 4개 차원을 1~5점 정수로 평가한다.

| 항목 | 설명 |
|---|---|
| `domain_match` | rationale이 해당 영역 기준에 맞는가 |
| `score_rationale_consistency` | predicted_score와 rationale 내용이 서로 부합하는가 |
| `specificity` | rationale이 에세이의 특정 문장·표현·논지를 구체적으로 짚는가 |
| `groundedness` | rationale이 실제 essay_text에 근거하는가 (환각 없는가) |
| `overall_judge` | 위 4개 점수의 가중 평균 (Python에서 직접 계산) |

### overall_judge 계산 규칙 (`compute_overall()`)

1. 4개 점수의 단순 평균
2. 최저점 기반 상한 적용:
   - 최저점 == 1 → 평균을 **2.0** 이하로 제한
   - 최저점 == 2 → 평균을 **3.0** 이하로 제한

> `legacy_rationale_single_judge.py`의 `compute_overall()`과 달리 `domain_match ≤ 2` / `groundedness ≤ 2` 상한(2.5) 규칙은 적용되지 않는다.

---

## 체크포인트 (중단 후 재개)

**아이템 단위** 체크포인트. 출력 파일에 이미 기록된 `essay_id`는 건너뛰고, 미처리 항목만 이어서 처리한다.  
중간에 중단해도 완료된 에세이부터 자동으로 재개된다.

---

## 사전 조건

- **GPT 사용 시**: `api_key`가 `IterConfig`(`src/essay/mad2_consensus_iter.py`)에 설정되어 있어야 한다.
- **Gemma / Qwen 사용 시**: LM Studio가 `http://localhost:1234`에서 실행 중이어야 하며, 해당 모델이 로드된 상태여야 한다.
