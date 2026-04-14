# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 실행 환경

- Python: `c:\Users\c\AppData\Local\Programs\Python\Python312\python.exe`
- 작업 디렉토리: `c:/Users/c/OneDrive/Desktop/NLP lab/제안서/말평/초기실험 코드/`
- 스크립트 실행 예시:
  ```
  c:\Users\c\AppData\Local\Programs\Python\Python312\python.exe "c:/Users/c/OneDrive/Desktop/NLP lab/제안서/말평/초기실험 코드/src/MAD3.py" --model gemma
  ```

## 프로젝트 개요

한국어 에세이 자동 채점 모델(Gemma, Qwen, LLaMA, GPT)의 **예측 결과를 자동으로 평가**하는 파이프라인. 평가 대상은 에세이 자체가 아니라 **모델이 생성한 점수(predicted_score)와 근거(rationale)**의 품질이다.

평가 영역은 3개: `content`, `organization`, `expression`  
평가 기준은 4개 차원: `domain_match`, `score_rationale_consistency`, `specificity`, `groundedness`

## 디렉토리 구조

```
초기실험 코드/
├── src/                                              스크립트
├── docs/                                             스크립트별 사용법 문서
├── data/selected_prompt_jsons/                       에세이 원본
├── inference_results/gemma|qwen|llama|gpt/           모델 예측 결과
├── judge_results/
│   ├── exp01_single/{judge_model}/{essay_model}/     single_judge.py 출력
│   ├── exp02_mad/{essay_model}/                      MAD.py 출력
│   ├── exp03_mad2/{essay_model}/                     MAD2.py 출력
│   ├── exp04_mad3/{essay_model}/                     MAD3.py 출력 (중첩)
│   ├── exp05_iter/{judge_model}/iter{N}/{essay_model}/  MAD3_iter.py 출력
│   └── exp06_crossmodel/{tag}/{essay_model}/         MAD3_crossmodel.py 출력 (미생성)
├── stats/
│   ├── inference_summary/{model}.json                모델별 추론 요약
│   └── *.json                                        집계 JSON
└── 실험결과 시각화/
```

## 스크립트 → 출력 디렉토리 매핑

| 스크립트 | 출력 디렉토리 | 구조 |
|---|---|---|
| `single_judge.py` | `judge_results/exp01_single/{judge_model}/{essay_model}/` | 평탄 |
| `MAD.py` | `judge_results/exp02_mad/{essay_model}/` | 평탄 |
| `MAD2.py` | `judge_results/exp03_mad2/{essay_model}/` | 평탄 (4개 점수 + overall_judge) |
| `MAD3.py` | `judge_results/exp04_mad3/{essay_model}/` | 중첩 (strict_initial/lenient_initial/strict_adjusted/lenient_adjusted/final) |
| `MAD3_iter.py` | `judge_results/exp05_iter/{judge_model}/iter{N}/{essay_model}/` | 중첩 (round_0…round_N-1/final) |
| `MAD3_crossmodel.py` | `judge_results/exp06_crossmodel/{tag}/{essay_model}/` | 중첩 (MAD3와 동일) |

## 핵심 스크립트

### 데이터 전처리
- `src/transform_json.py` — 원시 채점 데이터를 에세이 단위 JSON으로 변환 (paragraph 이어붙임, 5점 척도 정규화, 라벨 통합)
- `src/Q_classifier.py` — 프롬프트별 샘플을 단일 디렉토리로 추출 (`selected_prompt_jsons/` 생성)

### 추론 (에세이 채점)
- `src/inf_gpt.py` — GPT로 에세이 채점, `inference_results/gpt/` 출력
- `src/inf_ollama.py` — Ollama 로컬 모델(LLaMA 등)로 채점, `inference_results/{model}/` 출력
- `src/inf_lmstudion.py` — LM Studio 모델로 채점, `inference_results/{model}/` 출력

### 자동 평가 (Judge)
- `src/single_judge.py` — 단일 LLM이 rationale 품질 평가 (비교 기준선), 파일 단위 체크포인트
- `src/MAD.py` — 2-agent 토론 (Critic → Defender)
- `src/MAD2.py` — 3-agent 토론 (Critic → Defender → Final Judge)
- `src/MAD3.py` — Strict/Lenient 상호조정 방식, **아이템 단위 체크포인트** (중단 후 재개 가능)
- `src/MAD3_iter.py` — MAD3의 N회 반복 확장. `--iterations N` 인자로 교환 횟수 지정 (기본 3). `--judge-model` 생략 시 gpt→gemma→qwen 순차 실행
- `src/MAD3_crossmodel.py` — Strict/Lenient가 다른 모델을 사용. `--strict-model`, `--lenient-model` 등 지정

### 통계 집계
- `src/get_judge_score.py` — `overall_judge` 기준 통계 (exp03_mad2 호환)
- `src/get_judge2_score.py` — 4개 차원 점수 통계, `--output-prefix` 인자로 출력 파일명 제어, `stats/` 하위 저장
- `src/get_mad4_score.py` — exp04_mad3 전용. `final` 서브키에서 4개 점수 추출 → `stats/mad4_stats_all.json`
- `src/get_metrics.py` — Range + Distribution 메트릭 (CV, normalized entropy, mode_freq 등). `--exp` 또는 `--all`
- `src/get_score.py` — 예측 점수(predicted_score) 통계

### 집계 실행 명령
```
# MAD2 결과 집계 (exp03_mad2)
python src/get_judge2_score.py --output-prefix judge_MAD3

# MAD3 결과 집계 (exp04_mad3) → stats/mad4_stats_all.json
python src/get_mad4_score.py

# 전체 메트릭 집계 → stats/metrics_all.json
python src/get_metrics.py --all
```

### 새 실험 실행 명령
```
# Iteration 증가 (judge-model 생략 시 gpt→gemma→qwen 순차 실행)
python src/MAD3_iter.py --model gpt --iterations 3
python src/MAD3_iter.py --model gpt --iterations 5
python src/MAD3_iter.py  # judge-model 전체 × iter3+iter5 순차 실행

# Cross-model (GPT strict + Qwen instruct lenient)
python src/MAD3_crossmodel.py \
  --strict-model gpt-4o-mini \
  --lenient-model qwen2.5-7b-instruct \
  --lenient-url http://localhost:1234/v1 \
  --essay-model gemma \
  --exp-tag gpt_instruct9b
```

### 시각화
- `실험결과 시각화/mad_comparison_bar.py` — exp01/exp03/exp04 비교 bar chart. `stats/` 하위 JSON 필요
- `실험결과 시각화/mad4_mean_bar.py` — exp04_mad3 결과만 시각화
- `실험결과 시각화/model_comparison.py` — 모델별 예측 점수 비교
- `실험결과 시각화/cpu_gpu_time_trend.py` — CPU/GPU 처리 시간 추이 시각화

## 스크립트 사용법 문서 (`docs/`)

각 스크립트의 상세 사용법은 `docs/` 디렉토리에 별도 파일로 정리되어 있다.

| 문서 파일 | 대상 스크립트 |
|---|---|
| `docs/MAD3_iter_usage.md` | `src/MAD3_iter.py` |
| `docs/single_judge_usage.md` | `src/single_judge.py` |

**스크립트를 수정한 경우, 반드시 `docs/` 의 해당 문서도 함께 업데이트해야 한다.**  
변경된 인자, 기본값, 동작 방식, 실행 예시 등이 문서에 정확히 반영되어 있어야 한다.

## MAD3.py 구조 (가장 최신)

**3단계 파이프라인**, 도메인당 API 4회 호출:

1. `strict_judge()` + `lenient_judge()` — 독립 평가 (temperature=0.1)
2. `strict_adjust(lenient_initial)` + `lenient_adjust(strict_initial)` — 상호 참조 조정 (temperature=0.0)
3. Python에서 두 조정 결과의 평균 → `final`

출력 구조 (도메인별):
```json
{
  "strict_initial":   { "4개 점수", "overall_judge", "rationale_for_score" },
  "lenient_initial":  { "4개 점수", "overall_judge", "rationale_for_score" },
  "strict_adjusted":  { "4개 점수", "overall_judge", "adjustment_notes" },
  "lenient_adjusted": { "4개 점수", "overall_judge", "adjustment_notes" },
  "final":            { "4개 점수(float 평균)", "overall_judge" }
}
```

`overall_judge`는 LLM 응답값을 무시하고 `compute_overall()`로 재계산 (min_score==1 → cap 2.0, min_score==2 → cap 3.0).

> `single_judge.py`의 `compute_overall()`은 추가 규칙이 있음: `domain_match ≤ 2` 또는 `groundedness ≤ 2`이면 평균을 2.5로 추가 제한.

### 모델별 실행
```
python src/MAD3.py --model gemma   # judge_results/exp04_mad3/gemma/
python src/MAD3.py --model qwen    # judge_results/exp04_mad3/qwen/
python src/MAD3.py --model llama   # judge_results/exp04_mad3/llama/
python src/MAD3.py --model gpt     # judge_results/exp04_mad3/gpt/
```
`--model` 생략 시 4개 순차 실행.

## 데이터 포맷

**에세이 원본** (`data/selected_prompt_jsons/*.json`):
```json
{
  "essay_id": "GWGR2400103100.1",
  "prompt_text": "...",
  "essay_text": "...",
  "label_5scale_evaluator1": {"con": 3, "org": 3, "exp": 3},
  "label_5scale_evaluator2": {"con": 3, "org": 3, "exp": 3},
  "label_5scale_average": {"con": 3.0, "org": 3.0, "exp": 3.0}
}
```

**모델 예측** (`inference_results/{model}/`):
```json
{
  "essay_id": "...",
  "prediction": {
    "content":      {"score": 3.5, "rationale": "..."},
    "organization": {"score": 3.0, "rationale": "..."},
    "expression":   {"score": 4.0, "rationale": "..."}
  }
}
```

## API 설정

- 모든 스크립트의 `api_key`가 소스 코드에 하드코딩되어 있음 (OpenAI key).
- `Config` 클래스(MAD2/MAD3) 또는 상수(`API_KEY`)로 관리.
- `max_tokens`: MAD2=512, MAD3=1024 (한국어 텍스트 밀도 고려).
