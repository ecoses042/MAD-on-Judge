# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 실행 환경

- Python: `c:\Users\c\AppData\Local\Programs\Python\Python312\python.exe`
- 작업 디렉토리: `c:/Users/c/OneDrive/Desktop/NLP lab/제안서/말평/초기실험 코드/`
- 스크립트 실행 예시:
  ```
  c:\Users\c\AppData\Local\Programs\Python\Python312\python.exe "c:/Users/c/OneDrive/Desktop/NLP lab/제안서/말평/초기실험 코드/src/essay/MAD_A_iter.py" --judge-model gpt --model gemma --iterations 3
  ```

## 프로젝트 개요

한국어 에세이 자동 채점 모델(Gemma, Qwen, LLaMA, GPT)의 **예측 결과를 자동으로 평가**하는 파이프라인. 평가 대상은 에세이 자체가 아니라 **모델이 생성한 점수(predicted_score)와 근거(rationale)**의 품질이다.

평가 영역은 3개: `content`, `organization`, `expression`  
평가 기준은 4개 차원: `domain_match`, `score_rationale_consistency`, `specificity`, `groundedness`

## 디렉토리 구조

```
초기실험 코드/
├── src/
│   ├── env_utils.py                                  공유 유틸리티 (essay/summeval 모두 사용)
│   ├── essay/                                        한국어 에세이 파이프라인 스크립트
│   └── summeval/                                     SummEval 영어 요약 파이프라인 스크립트
├── docs/                                             스크립트별 사용법 문서
├── data/selected_prompt_jsons/                       에세이 원본
├── inference_results/gemma|qwen|llama|gpt/           모델 예측 결과
├── judge_results/
│   ├── single_judge/{judge_model}/{essay_model}/     single_judge.py 출력
│   ├── mad_c/{essay_model}/                          MAD_C.py 출력
│   ├── mad_a_iter/{judge_model}/iter{N}/{essay_model}/  MAD_A_iter.py 출력
│   └── legacy 결과 디렉토리들                         현재 outline 범위 밖
├── stats/
│   ├── inference_summary/{model}.json                모델별 추론 요약
│   └── *.json                                        집계 JSON
└── 실험결과 시각화/
```

## 스크립트 → 출력 디렉토리 매핑

| 스크립트 | 출력 디렉토리 | 구조 |
|---|---|---|
| `essay/single_judge.py` | `judge_results/single_judge/{judge_model}/{essay_model}/` | 평탄 |
| `essay/MAD_C.py` | `judge_results/mad_c/{essay_model}/` | 평탄 |
| `essay/MAD_A_iter.py` | `judge_results/mad_a_iter/{judge_model}/iter{N}/{essay_model}/` | 중첩 (round_0…round_N-1/final) |
| `summeval/single_judge_summeval.py` | `summeval_judge_results/single_judge/{judge_model}/{system_name}/` | 평탄 |
| `summeval/MAD_C_summeval.py` | `summeval_judge_results/mad_c/{judge_model}/{system_name}/` | 평탄 |
| `summeval/MAD_A_iter_summeval.py` | `summeval_judge_results/mad_a_iter/{judge_model}/iter{N}/{system_name}/` | 중첩 |

## 핵심 스크립트

### 데이터 전처리 (에세이)
- `src/essay/transform_json.py` — 원시 채점 데이터를 에세이 단위 JSON으로 변환
- `src/essay/Q_classifier.py` — 프롬프트별 샘플을 단일 디렉토리로 추출 (`selected_prompt_jsons/` 생성)

### 추론 (에세이 채점)
- `src/essay/inference_essay/inf_gpt.py` — GPT로 에세이 채점, `inference_results/gpt/` 출력
- `src/essay/inference_essay/inf_ollama.py` — Ollama 로컬 모델(LLaMA 등)로 채점
- `src/essay/inference_essay/inf_lmstudion.py` — LM Studio 모델로 채점

### 자동 평가 — 에세이 Judge
- `src/essay/single_judge.py` — 단일 LLM이 rationale 품질 평가 (비교 기준선)
- `src/essay/MAD_C.py` — 2-agent 토론 (Critic → Defender)
- `src/essay/MAD_A_iter.py` — MAD-A의 N회 반복 확장. `--iterations N` 인자로 교환 횟수 지정 (기본 3)
- `src/essay/legacy/` — 현재 `outline.md` 범위 밖이거나 구 이름 체계의 보관 스크립트

### 자동 평가 — SummEval Judge
- `src/summeval/prepare_summeval.py` — HF `mteb/summeval` 다운로드 → `summeval_judge_input/` 생성
- `src/summeval/single_judge_summeval.py` — Single judge (coherence/consistency/fluency/relevance)
- `src/summeval/MAD_C_summeval.py` — MAD-C (Critic → Defender → Final)
- `src/summeval/MAD_A_iter_summeval.py` — MAD-A iter N회 반복

### 통계 집계
- `src/essay/get_metrics.py` — 에세이 judge 실험 Range + Distribution 메트릭 집계
- `src/essay/get_score.py` — 예측 점수(predicted_score) 통계
- `src/summeval/get_metrics_summeval.py` — SummEval judge 메트릭 집계

legacy 집계 스크립트:
- `src/essay/legacy/get_judge_score.py`
- `src/essay/legacy/get_judge2_score.py`
- `src/essay/legacy/get_mad_a_base_score.py`

### 집계 실행 명령
```
# 에세이 전체 메트릭 집계 → stats/metrics_all.json
python src/essay/get_metrics.py --all

# 단일 조건 집계 예시
python src/essay/get_metrics.py --exp single_judge/gpt
python src/essay/get_metrics.py --exp mad_c
python src/essay/get_metrics.py --exp mad_a_iter/gpt/iter3

# SummEval 메트릭 집계 → stats/summeval_metrics_all.json
python src/summeval/get_metrics_summeval.py --all
```

### 새 실험 실행 명령
```
# 에세이 — Iteration 증가
python src/essay/MAD_A_iter.py --model gpt --iterations 3
python src/essay/MAD_A_iter.py --model gpt --iterations 5

# SummEval 전체 순서
python src/summeval/prepare_summeval.py --sample-size 100
python src/summeval/single_judge_summeval.py --judge-model gpt --workers 3
python src/summeval/MAD_C_summeval.py --judge-model gpt --workers 3
python src/summeval/MAD_A_iter_summeval.py --judge-model gpt --iterations 5 --workers 3
```

### 시각화
- `실험결과 시각화/model_comparison.py` — 모델별 예측 점수 비교
- `실험결과 시각화/cpu_gpu_time_trend.py` — CPU/GPU 처리 시간 추이 시각화
- `실험결과 시각화/legacy/` — 현재 outline 범위 밖 실험의 시각화 스크립트 보관

## 스크립트 사용법 문서 (`docs/`)

각 스크립트의 상세 사용법은 `docs/` 디렉토리에 별도 파일로 정리되어 있다.

| 문서 파일 | 대상 스크립트 |
|---|---|
| `docs/MAD_A_iter_usage.md` | `src/essay/MAD_A_iter.py` |
| `docs/single_judge_usage.md` | `src/essay/single_judge.py` |
| `docs/summeval_pipeline_usage.md` | `src/summeval/*.py` |

**스크립트를 수정한 경우, 반드시 `docs/` 의 해당 문서도 함께 업데이트해야 한다.**  
변경된 인자, 기본값, 동작 방식, 실행 예시 등이 문서에 정확히 반영되어 있어야 한다.

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
- `Config` 클래스(MAD-C seq/MAD-A) 또는 상수(`API_KEY`)로 관리.
- `max_tokens`: MAD-C seq=512, MAD-A=1024 (한국어 텍스트 밀도 고려).
