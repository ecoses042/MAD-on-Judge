# legacy_rationale_single_judge.py 사용 가이드

단일 LLM Judge가 에세이 채점 모델의 **predicted_score + rationale** 품질을 평가하는 스크립트.  
비교 실험의 기준선(baseline) 역할을 한다.

---

## 실행 명령

```bash
python src/essay/legacy_rationale_single_judge.py [--judge-model {gpt,gemma,qwen}]
```

### 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--judge-model` | `gpt` | Judge로 사용할 모델 |

### judge-model 선택지

| 값 | 실제 모델 | 백엔드 |
|---|---|---|
| `gpt` (기본) | `gpt-4o-mini` | OpenAI API |
| `gemma` | `google/gemma-3-4b` | LM Studio (`http://localhost:1234/v1`) |
| `qwen` | `qwen/qwen3.5-9b` | LM Studio (`http://localhost:1234/v1`) |

---

## 실행 예시

```bash
# GPT-4o-mini로 평가 (기본)
python src/essay/legacy_rationale_single_judge.py

# Gemma로 평가 (LM Studio 필요)
python src/essay/legacy_rationale_single_judge.py --judge-model gemma

# Qwen으로 평가 (LM Studio 필요)
python src/essay/legacy_rationale_single_judge.py --judge-model qwen
```

---

## 입출력 구조

### 입력

| 경로 | 내용 |
|---|---|
| `inference_results/gemma/` | Gemma 채점 결과 |
| `inference_results/gpt/` | GPT 채점 결과 |
| `inference_results/qwen/` | Qwen 채점 결과 |
| `inference_results/llama/` | LLaMA 채점 결과 |
| `data/selected_prompt_jsons/` | 에세이 원본 (prompt_text, essay_text). 현재 샘플 생성 스크립트의 기본 출력은 `data/selected_prompt_jsons_100/`이므로 실행 전 경로를 통일해야 한다. |

### 출력

```
judge_results/single_judge/{judge_model}/{essay_model}/
```

예시:

```
judge_results/single_judge/
├── gpt/
│   ├── gemma/
│   ├── gpt/
│   ├── qwen/
│   └── llama/
├── gemma/
│   ├── gemma/
│   ├── gpt/
│   ├── qwen/
│   └── llama/
└── qwen/
    ├── gemma/
    ├── gpt/
    ├── qwen/
    └── llama/
```

출력 파일명은 입력 파일명과 동일하다 (예: `prompt_001.json` → `prompt_001.json`).

---

## 출력 JSON 구조

```json
[
  {
    "essay_id": "GWGR2400103100.1",
    "source_id": "...",
    "prompt_id": "...",
    "status": "...",
    "gold": { ... },
    "prediction": {
      "content":      { "score": 3.5, "rationale": "..." },
      "organization": { "score": 3.0, "rationale": "..." },
      "expression":   { "score": 4.0, "rationale": "..." }
    },
    "judge": {
      "content": {
        "domain_match": 4,
        "score_rationale_consistency": 3,
        "specificity": 3,
        "groundedness": 4,
        "overall_judge": 3.5
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
| `domain_match` | rationale이 해당 영역(content/organization/expression) 기준에 맞는가 |
| `score_rationale_consistency` | predicted_score와 rationale 내용이 서로 부합하는가 |
| `specificity` | rationale이 에세이의 특정 문장·표현·논지를 구체적으로 짚는가 |
| `groundedness` | rationale이 실제 essay_text에 근거하는가 (환각 없는가) |
| `overall_judge` | 위 4개 점수의 가중 평균 (Python에서 직접 계산, LLM 응답값 무시) |

### overall_judge 계산 규칙

1. 4개 점수의 단순 평균
2. 치명적 결함 상한 적용:
   - `domain_match ≤ 2` 또는 `groundedness ≤ 2` → 평균을 **2.5** 이하로 제한
3. 최저점 기반 상한 적용:
   - 최저점 == 1 → 평균을 **2.0** 이하로 제한
   - 최저점 == 2 → 평균을 **3.0** 이하로 제한

---

## 체크포인트 (중단 후 재개)

출력 파일이 이미 존재하면 해당 파일을 건너뜀(`[SKIP]` 출력).  
중간에 중단해도 완료된 파일부터 자동으로 재개된다.  
단, **파일 단위** 체크포인트이므로 파일 내부에서 중단된 경우 해당 파일은 재처리된다.

---

## 사전 조건

- **GPT 사용 시**: `API_KEY`가 소스 코드(`src/essay/legacy_rationale_single_judge.py` 8번째 줄)에 설정되어 있어야 한다.
- **Gemma / Qwen 사용 시**: LM Studio가 `http://localhost:1234`에서 실행 중이어야 하며, 해당 모델이 로드된 상태여야 한다.
