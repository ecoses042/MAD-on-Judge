# SummEval 실험 구현 계획

> **기존 실험 결과(judge_results/, inference_results/, data/selected_prompt_jsons/, src/*.py)는 건드리지 않는다.**  
> SummEval 관련 파일은 별도 디렉토리에서 완전히 독립 관리한다.

---

## 디렉토리 구조 (추가분)

```
초기실험 코드/
├── data/summeval/                         ← SummEval 원본 데이터
│   ├── raw/                               ← HuggingFace 다운로드 캐시
│   └── articles_sampled.json             ← 100개 기사 샘플 (확정본)
├── summeval_judge_input/                  ← 에세이의 inference_results에 해당
│   └── {system_name}/                    ← 요약 시스템별 평가 입력 JSON
├── summeval_judge_results/                ← 에세이의 judge_results에 해당
│   ├── single_judge/{judge_model}/
│   ├── mad_c/
│   └── mad_a_iter/{judge_model}/iter{N}/
└── src/summeval/                          ← SummEval 전용 스크립트
    ├── prepare_summeval.py
    ├── summeval_judge_prompt_utils.py
    ├── single_judge_summeval.py
    ├── MAD_A_iter_summeval.py
    ├── MAD_C_summeval.py
    └── get_metrics_summeval.py
```

---

## SummEval vs 기존 에세이 데이터셋 비교

| 항목 | 에세이 | SummEval |
|------|--------|----------|
| 평가 대상 | LLM이 생성한 채점 점수+근거 | 요약 시스템의 요약문 자체 |
| 도메인(dimension) | content / organization / expression | coherence / consistency / fluency / relevance |
| "모델 예측"의 의미 | 채점 LLM의 score + rationale | 요약 시스템의 summary text |
| gold label | 인간 채점자 점수 (5점 척도) | 크라우드소싱 어노테이터 점수 (1~5) |
| inference 단계 필요 여부 | 필요 (채점 LLM 실행) | 불필요 (요약문이 곧 평가 대상) |
| 언어 | 한국어 | 영어 |

**핵심**: 에세이 실험에서 Judge는 "LLM이 생성한 rationale의 품질"을 평가했지만,  
SummEval에서는 **요약문 자체를 Judge가 직접 평가**한다. `inference_results` 단계가 없다.

---

## JSON 구조 설계

### A. 원본 데이터 (`data/summeval/articles_sampled.json`)

```json
{
  "article_id": "cnn_dm_00001",
  "article_text": "...(뉴스 기사 원문)...",
  "system_summaries": {
    "system_A": {
      "summary": "...",
      "human_scores": {
        "coherence":   {"annotator_avg": 4.0,  "annotators": [4, 5, 3]},
        "consistency": {"annotator_avg": 4.67, "annotators": [5, 4, 5]},
        "fluency":     {"annotator_avg": 3.67, "annotators": [4, 4, 3]},
        "relevance":   {"annotator_avg": 3.67, "annotators": [3, 4, 4]}
      }
    }
  }
}
```

### B. Judge 입력 (`summeval_judge_input/{system_name}/{article_id}.json`)

에세이의 `inference_results`에 해당하는 중간 표현.  
요약문을 기존 파이프라인과 동일한 인터페이스로 변환한다.

```json
{
  "item_id": "cnn_dm_00001__system_A",
  "article_id": "cnn_dm_00001",
  "system_name": "system_A",
  "article_text": "...",
  "summary_text": "...",
  "human_scores": {
    "coherence":   4.0,
    "consistency": 4.67,
    "fluency":     3.67,
    "relevance":   3.67
  },
  "dimensions": ["coherence", "consistency", "fluency", "relevance"]
}
```

### C. Judge 결과 (`summeval_judge_results/{method}/`)

에세이의 `judge_results`와 동일한 구조.

**Single Judge / MAD-C**:
```json
{
  "item_id": "cnn_dm_00001__system_A",
  "article_id": "cnn_dm_00001",
  "system_name": "system_A",
  "human_scores": { "coherence": 4.0, "consistency": 4.67, "fluency": 3.67, "relevance": 3.67 },
  "judge": {
    "coherence":   {"domain_match": 4, "score_rationale_consistency": 3, "specificity": 3, "groundedness": 4, "overall_judge": 3.5},
    "consistency": { ... },
    "fluency":     { ... },
    "relevance":   { ... }
  }
}
```

**MAD-A (round별 구조 유지)**:
```json
{
  "judge": {
    "coherence": {
      "round_0": {"strict": {...}, "lenient": {...}},
      "round_1": {"strict": {...}, "lenient": {...}},
      "final":   {"domain_match": 3.5, "score_rationale_consistency": 3.0, "specificity": 3.0, "groundedness": 4.0, "overall_judge": 3.4}
    }
  }
}
```

---

## 구현 단계

### Step 1: 데이터 준비 — `src/summeval/prepare_summeval.py`

- SummEval HuggingFace 데이터셋(`mteb/summeval`) 또는 원본 JSON 로드
- 100개 기사 샘플링 (요약 품질 분포 고려)
- `data/summeval/articles_sampled.json` 저장
- `summeval_judge_input/{system_name}/` 중간 표현 생성

### Step 2: Judge 프롬프트 — `src/summeval/summeval_judge_prompt_utils.py`

기존 `judge_prompt_utils.py` 구조를 참조하되 SummEval용 dimension 기준 교체:

| Dimension | 판단 기준 |
|-----------|---------|
| coherence | 문장 간 논리적 연결성, 문단 전개의 일관성 |
| consistency | 원문 사실과의 일관성 (hallucination 여부) |
| fluency | 문법 정확성, 유창성, 자연스러운 표현 |
| relevance | 원문의 핵심 내용 반영도, 불필요한 정보 포함 여부 |

Judge 평가 기준 4개(`domain_match / score_rationale_consistency / specificity / groundedness`)는 **그대로 유지**.  
`[target_dimension별 판단 기준]` 섹션만 위 내용으로 교체.

`predicted_score` 필드: SummEval에는 없으므로 `human_score_avg`를 채워 Judge에게 제공하거나,  
`score_rationale_consistency`를 "요약문의 논리/사실이 평가 방향과 일치하는가"로 재해석.

### Step 3: Single Judge — `src/summeval/single_judge_summeval.py`

- 기존 `single_judge.py` 구조 그대로 참조
- `essay_index` → `article_index` (article_text 로드)
- `DOMAINS = ["content", "organization", "expression"]` → `DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]`
- 출력: `summeval_judge_results/single_judge/{judge_model}/`

### Step 4: MAD-A — `src/summeval/MAD_A_iter_summeval.py`

- 기존 `MAD_A_iter.py` 구조 그대로 참조
- `IterConfig.input_dirs` → `summeval_judge_input/` 디렉토리
- `IterConfig.output_dirs()` → `summeval_judge_results/mad_a_iter/{judge_model}/iter{N}/`
- STRICT/LENIENT 프롬프트의 domain 기준 섹션만 SummEval용으로 교체

### Step 5: MAD-C — `src/summeval/MAD_C_summeval.py`

- 기존 `MAD_C.py` 구조 그대로 참조, 동일한 방식으로 수정

### Step 6: 메트릭 집계 — `src/summeval/get_metrics_summeval.py`

- 기존 `get_metrics.py` 참조
- `SCORE_KEYS` 동일 유지 (`domain_match` 등 Judge 기준 4개)
- `AREAS` → `["coherence", "consistency", "fluency", "relevance"]`
- `MODELS` → SummEval 요약 시스템 목록

---

## 설계 결정 사항

1. **Judge 기준 4개 재사용**: `domain_match / score_rationale_consistency / specificity / groundedness`를 SummEval에 그대로 적용한다. 이 일관성이 "진동/앵커링 현상이 도메인 독립적임"을 보여주는 논문 주장에 부합한다.

2. **샘플 크기**: 100 기사 × 16 시스템 = 1,600 item이 기본. API 비용 고려 시 100 기사 × 4 시스템 서브샘플(400 item)도 옵션.

3. **기존 코드 불변**: 기존 스크립트를 수정하지 않고, `src/summeval/` 하위에 독립 스크립트로 구현한다. 공통 유틸(`env_utils.py` 등)은 그대로 import해서 재사용.
