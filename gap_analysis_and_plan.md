# outline.md vs 현재 구현 Gap 분석 및 해결 계획

> **작성일**: 2026-04-13  
> **기준 문서**: `outline.md` (버전 0.1)  
> **분석 범위**: `src/`, `judge_results/`, `stats/`

---

## 1. 현재 구현 상태 요약

| 실험 | 스크립트 | 출력 디렉토리 | 파일 수 | 주요 필드 |
|------|----------|--------------|--------|---------|
| exp01_single | `single_judge.py` | `judge_results/exp01_single/` | 3,600개 ✅ | 4개 점수, overall_judge |
| exp02_mad (MAD-C 전신) | `MAD.py` | `judge_results/exp02_mad/` | 1,200개 ✅ | 4개 점수 |
| exp03_mad2 (MAD-C) | `MAD2.py` | `judge_results/exp03_mad2/` | 1,200개 ✅ | strict/lenient/final, **winner_side 없음** |
| exp04_mad3 (MAD-A base) | `MAD3.py` | `judge_results/exp04_mad3/` | **0개 ❌** | — |
| exp05_iter iter=3 | `MAD3_iter.py` | `judge_results/exp05_iter/gpt/iter3/` | 3,053개 ✅ | round_0~N, final, adjustment_notes |
| exp05_iter iter=5 | `MAD3_iter.py` | `judge_results/exp05_iter/gpt/iter5/` | **미확인 ⚠️** | — |
| exp06_crossmodel | `MAD3_crossmodel.py` | `judge_results/exp06_crossmodel/` | 0개 (outline 외) | — |

---

## 2. outline.md 대비 Gap 목록

### Gap 1 — exp04_mad3 미실행 【Critical, ★★★】

- **현상**: `judge_results/exp04_mad3/` 하위에 JSON 파일 없음
- **영향**: RQ1(실험 1)에서 Single / MAD-C / MAD-A 3방식 비교 불가
  - MAD-A(iter=3)의 baseline은 exp04_mad3이어야 함
  - exp05_iter(iter=3)와 혼용할 경우 실험 설계 불명확
- **해결**: `MAD3.py` 4개 모델 전체 실행

---

### Gap 2 — MAD2.py 출력에 `winner_side` 누락 【Important, ★★☆】

- **현상**: `exp03_mad2` JSON에 `winner_side` 키 없음
- **outline 요구사항**: 방식 2(MAD-C) 출력에 `winner_side: critic|defender|tie` 기록 (outline.md line 112)
- **영향**: 보조 실험 5 (Defender 편향 분석) 전체 불가
  - `defender_win_rate` 계산 불가
  - `critic_major_issue_count` vs `final_score` 상관 분석 불가
- **해결**: MAD2.py 수정 + exp03_mad2 재실행 (또는 기존 출력에서 역산 가능 여부 검토)

> **역산 가능 여부**: 현재 exp03_mad2 출력에는 `strict_initial`, `lenient_initial` 점수는 있으나  
> critic/defender 개별 점수가 아닌 strict/lenient 구조라 직접 역산 불가.  
> MAD2.py는 Critic → Defender → Final Judge 3-agent 구조이므로,  
> 각 에이전트 출력을 별도로 저장해야 winner_side 계산 가능.

---

### Gap 3 — iter=5 데이터 완성도 미확인 【Important, ★★☆】

- **현상**: `exp05_iter/gpt/iter5/` 디렉토리 존재하나 파일 수 미확인
  - `gemma/iter5/`, `qwen/iter5/` 상태도 불명확
- **영향**: RQ4(실험 4) — iter=3 vs iter=5 수렴 패턴 비교 불가
  - `stabilization_round`, `flip_count_iter3 vs flip_count_iter5` 비교 불가
- **해결**: 파일 수 확인 후 부족 시 보충 실행

---

### Gap 4 — adjustment_notes 텍스트 구조화 미비 【Minor, ★☆☆】

- **현상**: `adjustment_notes`가 자유 텍스트로만 저장됨
  ```
  "adjustment_notes": "domain_match는 관대한 심사위원과 동일하게 근거의 깊이가 부족하다는..."
  ```
- **영향**: RQ3(실험 3) — anchor_ratio 계산 시 regex 기반 분석 필요 (정확도 저하 위험)
- **해결 선택지**:
  - A. 현 상태로 진행 → regex/키워드 매칭으로 anchor_pattern/logic_pattern 추출 (비용 없음)
  - B. MAD3_iter.py 수정하여 LLM에 구조화 응답 요청 (API 비용 증가)
- **권장**: Phase 1 완료 후 텍스트 샘플 검토 → A 방식으로 충분하면 유지

---

### Gap 5 — exp06_crossmodel 미실행 【Low, ★☆☆】

- **현상**: `MAD3_crossmodel.py` 존재하나 실행 결과 없음
- **영향**: outline.md의 RQ1~5에는 포함되지 않음 (CLAUDE.md 에만 기술)
- **해결**: outline.md 범위 외이므로 우선순위 낮음. 주요 실험 완료 후 선택 실행

---

## 3. 단계별 해결 계획

### Phase 1: 데이터 확보 (최우선)

#### Step 1-1. iter=5 완성도 확인
```bash
# judge_results/exp05_iter 내 iter5 파일 수 확인
find "judge_results/exp05_iter" -name "*.json" | grep "iter5" | wc -l
find "judge_results/exp05_iter" -name "*.json" | grep "iter5" | head -5
```
- **판단 기준**: essay 모델별 예상 파일 수 = 300개 (에세이 수)
- **완성 기준**: gpt/gemma/qwen × iter5 × 4 essay 모델 = 3,600개 이상

#### Step 1-2. exp04_mad3 실행
```bash
python src/MAD3.py  # --model 생략 시 4개 모델 순차 실행
```
- **예상 소요**: ~8시간 (300 essays × 4 models × 3 domains × 4 API calls)
- **체크포인트**: 아이템 단위 재개 가능 (중단 후 재실행 OK)

#### Step 1-3. iter=5 보충 실행 (미완성인 경우)
```bash
python src/MAD3_iter.py --judge-model gpt --iterations 5
python src/MAD3_iter.py --judge-model gemma --iterations 5
python src/MAD3_iter.py --judge-model qwen --iterations 5
```

---

### Phase 2: MAD2.py 수정 — winner_side 추가 (Codex에 위임)

**수정 범위**: `src/MAD2.py`

**구현 사양**:

1. `final_judge()` 호출 후, Critic과 Defender의 4개 항목별 점수 비교
2. 각 항목별 `winner_side` 결정 로직:
   ```python
   # 각 항목(domain_match 등)별로
   for key in ["domain_match", "score_rationale_consistency", "specificity", "groundedness"]:
       c = critic_score[key]
       d = defender_score[key]
       winner_side[key] = "critic" if c > d else ("defender" if d > c else "tie")
   ```
3. 출력 JSON에 `winner_side` 필드 추가:
   ```json
   {
     "judge": {
       "content": {
         ...,
         "winner_side": {
           "domain_match": "defender",
           "score_rationale_consistency": "tie",
           "specificity": "defender",
           "groundedness": "critic"
         }
       }
     }
   }
   ```

**주의**: exp03_mad2 기존 데이터는 재실행하거나, Critic/Defender 점수가 현재 출력에 포함되어 있다면 후처리 스크립트로 보완 가능.

> MAD2.py 출력 구조 재확인 필요: `strict_initial`/`lenient_initial`이 Critic/Defender에 대응하는지 검토.  
> 대응 관계가 맞다면 기존 파일 후처리로 winner_side 계산 가능 (API 재호출 불필요).

---

### Phase 3: 분석 스크립트 구현 (Codex에 위임)

outline.md 각 RQ에 대응하는 분석 스크립트 신규 작성:

| 스크립트 | 담당 RQ | 입력 | 출력 |
|---------|---------|------|------|
| `src/analyze_rq1_distribution.py` | RQ1 | exp01~exp04 final 점수 | Range/Std/IQR 비교, violin plot |
| `src/analyze_rq2_oscillation.py` | RQ2 | exp05_iter round별 데이터 | flip_count 분포, delta 방향 |
| `src/analyze_rq3_anchoring.py` | RQ3 | adjustment_notes 텍스트 | anchor_ratio, logic_ratio |
| `src/analyze_rq4_convergence.py` | RQ4 | iter3 vs iter5 | stabilization_round, range 비교 |
| `src/analyze_rq5_defender.py` | RQ5 보조 | exp03_mad2 winner_side | defender_win_rate |

---

## 4. 우선순위 요약

| 우선순위 | 작업 | 방법 | 의존성 |
|---------|------|------|--------|
| ★★★ | exp04_mad3 실행 | 실행 명령 | API 비용 |
| ★★★ | iter=5 완성도 확인 | bash 확인 | — |
| ★★☆ | MAD2.py winner_side 추가 | Codex 위임 | — |
| ★★☆ | 분석 스크립트 작성 (RQ1~4) | Codex 위임 | Phase 1 완료 후 |
| ★☆☆ | adjustment_notes 구조화 여부 결정 | 텍스트 샘플 검토 | Phase 1 완료 후 |
| ★☆☆ | exp06_crossmodel 실행 | 선택사항 | outline 외 |

---

## 5. 실험 로직 재매핑 (outline.md ↔ 실제 디렉토리)

| outline.md 방식 | 실제 디렉토리 | 비고 |
|----------------|-------------|------|
| Single Judge (baseline) | `exp01_single/` | ✅ 완성 |
| 비판 MAD (MAD-C) | `exp03_mad2/` | ⚠️ winner_side 누락 |
| 합의 MAD iter=3 (MAD-A) | `exp04_mad3/` | ❌ 미실행 |
| 합의 MAD iter=5 | `exp05_iter/*/iter5/` | ⚠️ 완성도 미확인 |

> **주의**: `exp02_mad/`(MAD.py)는 outline.md의 비판 MAD와 구조가 다름 (2-agent, Critic→Defender만 있고 Final Judge 없음).  
> outline.md의 MAD-C = `exp03_mad2/`(MAD2.py) 로 보는 것이 적합.

---

## 6. 현재 즉시 실행 가능한 분석

Phase 1 없이도 아래 분석은 현재 데이터로 가능:

- **RQ1 부분 분석**: exp01_single vs exp03_mad2 (2방식 비교, MAD-A 없이)
- **RQ2 완전 분석**: exp05_iter/iter3 데이터로 flip_count, delta 계산 가능
- **RQ3-A 완전 분석**: exp05_iter adjustment_notes 텍스트 분석 가능
- **RQ4 부분 분석**: iter3만으로 수렴 패턴 분석 가능 (iter5 비교 제외)
