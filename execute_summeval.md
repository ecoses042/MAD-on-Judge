# SummEval 실행 정리

## 진행사항

- `src/summeval/summeval_judge_prompt_utils.py`를 추가해 SummEval 전용 prompt 세트를 만들었다.
- `src/summeval/prepare_summeval.py`를 추가해 SummEval 원본/로컬 JSON 데이터를 정규화하고 `summeval_judge_input/`을 생성하도록 했다.
- `src/summeval/single_judge_summeval.py`를 추가해 SummEval 단일 Judge 평가를 수행하도록 했다.
- `src/summeval/MAD_C_summeval.py`를 추가해 Critic → Defender → Final Judge 구조의 MAD-C 평가를 수행하도록 했다.
- `src/summeval/MAD_A_iter_summeval.py`를 추가해 round 기반 MAD-A 반복 평가를 수행하도록 했다.
- 기존 에세이용 `src/*.py` 파일은 수정하지 않았다.

## 로직 흐름

1. `prepare_summeval.py`
   - 입력 JSON/JSONL/디렉토리에서 article 단위 record를 읽는다.
   - `article_id`, `article_text`, `system_summaries` 구조로 정규화한다.
   - 지정한 개수만큼 article을 샘플링한다.
   - `data/summeval/articles_sampled.json`에 샘플을 저장한다.
   - 각 `article_id × system_name` 조합에 대해 `summeval_judge_input/{system_name}/{article_id}.json`을 만든다.

2. `single_judge_summeval.py`
   - `summeval_judge_input/{system_name}/` 아래 파일을 읽는다.
   - 각 item에 대해 `coherence / consistency / fluency / relevance`를 순차 평가한다.
   - Judge 결과는 `summeval_judge_results/single_judge/{judge_model}/{system_name}/`에 저장한다.

3. `MAD_C_summeval.py`
   - 각 item에 대해 Critic과 Defender를 먼저 돌린다.
   - 두 결과를 Final Judge가 비교하여 최종 점수를 만든다.
   - 출력은 `summeval_judge_results/mad_c/{judge_model}/{system_name}/`에 저장한다.

4. `MAD_A_iter_summeval.py`
   - round 0에서 Strict/Lenient을 독립 평가한다.
   - 이후 round부터는 상대방의 직전 결과를 보고 각자 점수를 조정한다.
   - 마지막 round의 strict/lenient 결과를 평균해 `final` 점수를 만든다.
   - 출력은 `summeval_judge_results/mad_a_iter/{judge_model}/iter{N}/{system_name}/`에 저장한다.

## 핵심 설계

- Judge 기준은 기존 에세이 실험의 4개 항목 구조를 유지했다.
- SummEval 전용 dimension은 `coherence / consistency / fluency / relevance`로 교체했다.
- local LM Studio 모델(`gemma`, `qwen`)을 우선 지원하고, `gpt`는 fallback으로 남겼다.
- 각 스크립트는 JSON 복구, 체크포인트, resume 가능한 저장 방식을 유지했다.
- `python3 -m py_compile`로 새로 추가한 SummEval 스크립트들의 문법을 확인했다.
