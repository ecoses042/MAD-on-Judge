# MAD_A_iter_text_only.py 사용 가이드

`src/MAD_A_iter_text_only.py`는 MAD-A iteration 실험의 text-only ablation 버전이다.  
Strict/Lenient Judge의 **추론 실행 방식은 `src/MAD_A_iter.py`와 동일**하며, 각 라운드마다 strict/lenient **2개 추론을 병렬 실행**한다. 차이는 조정 라운드에서 상대 Judge의 **점수 숫자는 숨기고 텍스트 피드백만 공유**한다는 점이다.

---

## 실행 명령

```bash
python src/MAD_A_iter_text_only.py [--model {gemma,qwen,llama,gpt}] [--judge-model {gpt,gemma,qwen}] [--iterations N]
```

### 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--model` | 없음 (전체 순차 실행) | 처리할 에세이 모델 |
| `--judge-model` | 없음 (gpt → gemma → qwen 순서대로 실행) | Judge로 사용할 모델 |
| `--iterations` | 없음 (5회) | 총 교환 횟수 (최소 2) |

`--model`을 생략하면 기본적으로 모든 에세이 모델을 순차 처리한다. 단, `--judge-model qwen`과 함께 `--model`을 생략한 경우에는 `llama`, `gpt` 에세이 모델만 처리한다.

---

## 실행 예시

```bash
# 모든 인자 생략: gpt→gemma→qwen judge 조합으로 iter5 전체 순차 실행
python src/MAD_A_iter_text_only.py

# GPT judge, gpt 에세이 모델만, 5회
python src/MAD_A_iter_text_only.py --judge-model gpt --model gpt --iterations 5

# Gemma judge (LM Studio), 5회
python src/MAD_A_iter_text_only.py --judge-model gemma --iterations 5

# Qwen judge (LM Studio), --model 생략 시 llama/gpt 에세이 모델만 처리
python src/MAD_A_iter_text_only.py --judge-model qwen --iterations 5
```

---

## 라운드 구조

```
round_0: strict(독립 평가, temp=0.1)  ||  lenient(독립 평가, temp=0.1)
round_1: strict(lenient의 텍스트 피드백만 참조, temp=0.0)  ||  lenient(strict의 텍스트 피드백만 참조, temp=0.0)
...
round_N-1: strict(lenient_N-2의 텍스트 피드백만 참조)  ||  lenient(strict_N-2의 텍스트 피드백만 참조)
final: round_N-1의 strict + lenient 평균
```

- 각 라운드에서 strict/lenient 추론은 항상 **2개씩 병렬 실행**된다.
- `round_0`는 독립 평가이고, `round_1` 이후만 text-only 조정이 적용된다.
- 도메인당 API 호출 횟수는 `N × 2`회로 `MAD_A_iter.py`와 동일하다.

---

## MAD_A_iter.py와의 차이

| 항목 | `MAD_A_iter.py` | `MAD_A_iter_text_only.py` |
|---|---|---|
| strict/lenient 실행 수 | 라운드당 2개 병렬 | 라운드당 2개 병렬 |
| 조정 라운드 입력 | 상대 Judge의 전체 JSON | 상대 Judge의 텍스트 필드만 추출 |
| 숨기는 정보 | 없음 | 점수 필드(`domain_match`, `specificity` 등) |
| 출력 경로 | `judge_results/mad_a_iter/...` | `judge_results/mad_a_text_only/...` |

text-only 버전에서 상대에게 공유되는 필드는 아래 두 텍스트 항목만 사용한다.

```python
TEXT_FIELDS = ("rationale_for_score", "adjustment_notes")
```

---

## 입출력 구조

출력 경로:

```
judge_results/mad_a_text_only/{judge_model}/iter{N}/{essay_model}/
```

출력 JSON 구조와 `final` 계산 방식은 `MAD_A_iter.py`와 동일하다.

---

## 사전 조건

- **GPT 사용 시**: `OPENAI_API_KEY`가 설정되어 있어야 한다.
- **Gemma / Qwen 사용 시**: LM Studio가 `http://localhost:1234`에서 실행 중이어야 하며, 해당 모델이 로드된 상태여야 한다.
