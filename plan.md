# 디렉토리 구조 최적화 + 추가 실험 준비 계획

## Context

현재 프로젝트 루트에 15개 이상의 결과 폴더가 flat하게 쌓여 있어 실험 유형 구분이 어렵고, 새 실험(iteration 증가, cross-model debater)을 추가하면 더 복잡해진다. 추가 실험 시작 전 계층적 구조로 재편하고, 새 스크립트 설계까지 함께 준비한다.

**연구 질문**: 주관적 task(한국어 에세이 채점)에서 LLM Judge의 점수 일률화 문제와 인간 평가자와의 일관성 분석
- Debate iteration 수 증가 → 수렴 여부 확인
- Cross-model debater → 모델 조합에 따른 점수 분포 변화

---

## Phase 1: 디렉토리 구조 재편

### 목표 구조

```
초기실험 코드/
├── src/                               (스크립트 — 경로만 수정)
├── data/
│   └── selected_prompt_jsons/        ← MOVE from root
├── inference_results/                 ← 모델 예측 결과 통합
│   ├── gpt/                          ← MOVE from gpt_results/
│   ├── gemma/                        ← MOVE from results_google_gemma-3-4b/
│   ├── llama/                        ← MOVE from results_llama3_8b/
│   └── qwen/                         ← MOVE from results_qwen_qwen3.5-9b/
├── judge_results/
│   ├── exp01_single/                  ← MOVE from judge_single_results_*/ & judge_multi_results_*/
│   │   ├── gemma/ gpt/ llama/ qwen/
│   ├── exp02_mad/                     ← MOVE from mad_results_*/
│   │   ├── gemma/ gpt/ llama/ qwen/
│   ├── exp03_mad2/                    ← MOVE from mad2_results_*/ or mad3_results_*/
│   │   ├── gemma/ gpt/ llama/ qwen/
│   ├── exp04_mad3/                    ← MOVE from mad4_results_*/
│   │   ├── gemma/ gpt/ llama/ qwen/
│   ├── exp05_iter/                    ← NEW: iteration 증가 실험
│   │   ├── iter3/
│   │   │   ├── gemma/ gpt/ llama/ qwen/
│   │   └── iter5/
│   │       ├── gemma/ gpt/ llama/ qwen/
│   └── exp06_crossmodel/             ← NEW: cross-model debater
│       ├── gpt_baseline9b/            ← GPT(strict) + Baseline 9b(lenient)
│       │   ├── gemma/ llama/ qwen/    (essay model별 서브폴더)
│       ├── gpt_instruct9b/
│       └── baseline9b_instruct9b/
├── stats/                             ← MOVE aggregated JSONs from root
│   ├── judge_single_all.json
│   ├── judge_MAD3_all.json
│   └── mad4_stats_all.json
└── 실험결과 시각화/                   (그대로 유지)
```

### 이동할 파일/폴더 목록

| 현재 위치 | 이동 후 위치 |
|---|---|
| `selected_prompt_jsons/` | `data/selected_prompt_jsons/` |
| `gpt_results/` | `inference_results/gpt/` |
| `results_google_gemma-3-4b/` | `inference_results/gemma/` |
| `results_llama3_8b/` | `inference_results/llama/` |
| `results_qwen_qwen3.5-9b/` | `inference_results/qwen/` |
| `judge_single_results_*/` (또는 `judge_multi_results_*/`) | `judge_results/exp01_single/{model}/` |
| `mad_results_*/` | `judge_results/exp02_mad/{model}/` |
| `mad2_results_*/` | `judge_results/exp03_mad2/{model}/` |
| `mad3_results_*/` | `judge_results/exp03_mad2/{model}/` (중복 시 확인) |
| `mad4_results_*/` | `judge_results/exp04_mad3/{model}/` |
| `judge_single_all.json` | `stats/judge_single_all.json` |
| `judge_MAD3_all.json` | `stats/judge_MAD3_all.json` |
| `mad4_stats_all.json` | `stats/mad4_stats_all.json` |
| `results/` (요약 JSON 4개) | `stats/inference_summary/` |

---

## Phase 2: 스크립트 경로 수정

모든 스크립트는 프로젝트 루트 기준 상대 경로를 사용하므로, Config/상수만 수정하면 됨.

### `src/MAD3.py` — Config 수정
```python
essay_data_dir: str = "data/selected_prompt_jsons"
input_dirs: list[str] = field(default_factory=lambda: [
    "inference_results/gemma",
    "inference_results/qwen",
    "inference_results/llama",
    "inference_results/gpt",
])
output_dirs: list[str] = field(default_factory=lambda: [
    "judge_results/exp04_mad3/gemma",
    "judge_results/exp04_mad3/qwen",
    "judge_results/exp04_mad3/llama",
    "judge_results/exp04_mad3/gpt",
])
```

### `src/MAD2.py` — 동일 방식, exp03_mad2/

### `src/single_judge.py` — 상수 수정
```python
ESSAY_DATA_DIR = "data/selected_prompt_jsons"
INPUT_DIRS  = ["inference_results/gemma", "inference_results/gpt", ...]
OUTPUT_DIRS = ["judge_results/exp01_single/gemma", ...]
```

### `src/MAD.py` — 동일 방식, exp02_mad/

### `src/get_mad4_score.py` — INPUT/OUTPUT 경로
```python
INPUT_DIRS  = ["judge_results/exp04_mad3/gemma", ..., "judge_results/exp04_mad3/gpt"]
OUTPUT_FILE = "stats/mad4_stats_all.json"
```

### `src/get_judge2_score.py`, `src/get_judge_score.py`
- 입력 디렉토리 패턴을 `judge_results/exp03_mad2/*/` 로 변경
- 출력을 `stats/` 하위로 변경

### `src/inf_gpt.py`, `src/inf_ollama.py`, `src/inf_lmstudion.py`
- 출력 디렉토리를 `inference_results/{model}/` 로 변경

### `실험결과 시각화/*.py`
- JSON 로드 경로를 `../stats/` 기준으로 수정

---

## Phase 3: 새 실험 스크립트

### 3-A. `src/MAD3_iter.py` — Debate Iteration 증가

**핵심 변경**: MADPipeline이 N회 교환을 반복.

현재 MAD3 = 2회 교환:
```
strict_initial, lenient_initial  → Round 1 (independent)
strict_adjusted, lenient_adjusted → Round 2 (each sees other's Round 1)
final = avg(strict_adjusted, lenient_adjusted)
```

iter=N 구조:
```
rounds[0]: strict_r0, lenient_r0  (독립, temp=0.1)
rounds[1]: strict_r1 (sees lenient_r0), lenient_r1 (sees strict_r0)  (temp=0.0)
...
rounds[N-1]: strict_rN, lenient_rN  (each sees other's round N-2)
final = avg(strict_rN, lenient_rN)
```

출력 구조 변경:
```json
{
  "round_0": {"strict": {...}, "lenient": {...}},
  "round_1": {"strict": {...}, "lenient": {...}},
  "round_N": {"strict": {...}, "lenient": {...}},
  "final":   {"4개 점수(float 평균)", "overall_judge"}
}
```

**CLI 인터페이스**:
```bash
python src/MAD3_iter.py --model gemma --iterations 3
python src/MAD3_iter.py --model gemma --iterations 5
```

**출력 위치**: `judge_results/exp05_iter/iter{N}/{model}/`

**Config 추가 필드**:
```python
n_iterations: int = 3  # 총 교환 횟수 (2 = 기존 MAD3과 동일)
```

**재사용**: MAD3.py의 `strict_judge()`, `lenient_judge()`, `strict_adjust()`, `lenient_adjust()`, `compute_overall()`, `build_essay_index()`, `append_result()`, `load_checkpoint()` 그대로 사용.

### 3-B. `src/MAD3_crossmodel.py` — Cross-Model Debater

**핵심 변경**: strict judge와 lenient judge가 서로 다른 모델/API 사용.

두 개의 클라이언트 필요:
- `strict_client`: OpenAI API (GPT-4o-mini)
- `lenient_client`: LM Studio OpenAI-compatible API (`base_url="http://localhost:1234/v1"`)

**Config 구조**:
```python
@dataclass
class CrossModelConfig:
    # Strict judge 설정
    strict_api_key: str = "sk-..."           # OpenAI key
    strict_model: str = "gpt-4o-mini"
    strict_base_url: str = None              # None = 기본 OpenAI endpoint
    
    # Lenient judge 설정
    lenient_api_key: str = "lm-studio"       # LM Studio는 아무 값이나 가능
    lenient_model: str = "qwen2.5-7b-instruct"
    lenient_base_url: str = "http://localhost:1234/v1"
    
    essay_data_dir: str = "data/selected_prompt_jsons"
    input_dirs: list[str] = ...
    output_dirs: list[str] = ...
```

**MADPipeline 분리**: 현재 `api_client` 하나 → `strict_client`, `lenient_client` 두 개.

**CLI 인터페이스**:
```bash
# GPT(strict) + Baseline Qwen 9b(lenient) — essay model gemma 채점 결과 judge
python src/MAD3_crossmodel.py \
  --strict-model gpt-4o-mini \
  --lenient-model qwen2.5-7b \
  --lenient-url http://localhost:1234/v1 \
  --essay-model gemma \
  --exp-tag gpt_baseline9b
```

**출력 위치**: `judge_results/exp06_crossmodel/{exp_tag}/{essay_model}/`

**3개 실험 조합**:
```bash
# 1. GPT + Baseline 9b
--exp-tag gpt_baseline9b  --strict gpt-4o-mini  --lenient qwen2.5-7b

# 2. GPT + Instruct 9b  
--exp-tag gpt_instruct9b  --strict gpt-4o-mini  --lenient qwen2.5-7b-instruct

# 3. Baseline 9b + Instruct 9b
--exp-tag baseline9b_instruct9b  --strict qwen2.5-7b  --lenient qwen2.5-7b-instruct
```

---

## Phase 4: 새 메트릭 — `src/get_metrics.py`

### 4-A. Range 메트릭 (점수 다양성 측정)
```python
score_range = max(scores) - min(scores)          # 점수 분산 범위
iqr = Q75 - Q25                                  # 이상치 강건한 spread
```

### 4-B. Distribution 메트릭 (다른 range도 비교 가능)
점수 range가 모델/방법마다 달라도 비교 가능한 정규화 메트릭:

```python
# 1. Coefficient of Variation (CV): std/mean — range 중립적
cv = std / mean

# 2. Normalized Entropy: 정수 점수 분포의 균등성
#    score 1~5 → 빈도 분포 → Shannon entropy / log(n_bins)
from collections import Counter
import math
counts = Counter(round(s) for s in scores)
probs  = [c / len(scores) for c in counts.values()]
entropy = -sum(p * math.log2(p) for p in probs if p > 0)
n_entropy = entropy / math.log2(5)  # 5점 척도 기준 정규화

# 3. Score Concentration Index (점수 쏠림 측정)
#    특정 점수(3점)에 몰리는 정도
mode_freq = max(counts.values()) / len(scores)

# 4. Wasserstein Distance from Uniform (분포 균등성)
from scipy.stats import wasserstein_distance
uniform = [1/5] * 5  # 1~5점 균등 분포
actual  = [counts.get(i, 0)/len(scores) for i in range(1, 6)]
wd = wasserstein_distance(range(1,6), range(1,6), actual, uniform)
```

**출력 구조** (`stats/metrics_all.json`):
```json
{
  "exp04_mad3": {
    "gemma": {
      "domain_match": {
        "mean": 3.62, "std": 0.71, "range": 4.0, "iqr": 1.0,
        "cv": 0.196, "normalized_entropy": 0.72, "mode_freq": 0.45,
        "wasserstein_from_uniform": 0.38
      },
      ...
    }
  },
  "exp05_iter/iter3": { ... },
  "exp06_crossmodel/gpt_baseline9b": { ... }
}
```

### `src/get_metrics.py` CLI
```bash
# 단일 실험 집계
python src/get_metrics.py --exp exp04_mad3

# 전체 비교
python src/get_metrics.py --all
```

---

## Phase 5: 시각화 확장 (`실험결과 시각화/`)

- `iteration_convergence.py` — round별 점수 변화 line plot (수렴 확인)
- `crossmodel_distribution.py` — 모델 조합별 CV/entropy 비교 bar chart
- `distribution_heatmap.py` — 점수 분포를 heatmap (score bin × experiment)

---

## 실행 순서

```bash
# 1. 디렉토리 이동 (OS 명령으로 일괄)
# 2. 스크립트 경로 수정
# 3. 기존 실험 재집계 확인
python src/get_mad4_score.py
python src/get_judge2_score.py --output-prefix judge_MAD3

# 4. 새 실험 — Iteration study
python src/MAD3_iter.py --model gpt --iterations 3
python src/MAD3_iter.py --model gpt --iterations 5

# 5. 새 실험 — Cross-model
python src/MAD3_crossmodel.py --strict gpt-4o-mini --lenient qwen2.5-7b --essay-model gemma --exp-tag gpt_baseline9b

# 6. 통합 메트릭 집계
python src/get_metrics.py --all
```

---

## 수정 대상 파일 정리

| 파일 | 변경 내용 |
|---|---|
| `src/MAD3.py` | Config 경로 4개 수정 |
| `src/MAD2.py` | Config 경로 수정 |
| `src/single_judge.py` | 상수 3개 수정 |
| `src/MAD.py` | 경로 수정 |
| `src/get_mad4_score.py` | INPUT_DIRS, OUTPUT_FILE 수정 |
| `src/get_judge2_score.py` | 입/출력 경로 수정 |
| `src/get_judge_score.py` | 경로 수정 |
| `src/inf_gpt.py` | 출력 디렉토리 수정 |
| `src/inf_ollama.py` | 출력 디렉토리 수정 |
| `src/inf_lmstudion.py` | 출력 디렉토리 수정 |
| `실험결과 시각화/*.py` | JSON 로드 경로 수정 |
| `CLAUDE.md` | 새 구조 반영하여 업데이트 |
| **(NEW)** `src/MAD3_iter.py` | MAD3 iteration 확장 |
| **(NEW)** `src/MAD3_crossmodel.py` | Cross-model debater |
| **(NEW)** `src/get_metrics.py` | Range + Distribution 메트릭 |
| **(NEW)** `실험결과 시각화/iteration_convergence.py` | Iteration 시각화 |
| **(NEW)** `실험결과 시각화/crossmodel_distribution.py` | Cross-model 시각화 |
