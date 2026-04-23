# prepare_nikl_essay_dataset.py

국립국어원 2024년 글쓰기 채점 말뭉치 원본 JSON에서 `outline.md`의 에세이 설정에 맞춰 Q4~Q9 주제별 100개, 총 600개 샘플을 랜덤 추출한다.

## 실행

```bash
python src/essay/prepare_nikl_essay_dataset.py
```

기본 입력은 `data/NIKL_GRADING WRITING DATA 2024`, 기본 출력은 `data/selected_prompt_jsons_100`이다. 출력 파일은 기존 `data/selected_prompt_jsons` 예시와 동일하게 샘플 1개를 `[sample]` 형태의 JSON 리스트로 저장한다.

## 주요 옵션

```bash
python src/essay/prepare_nikl_essay_dataset.py \
  --raw-data-dir "data/NIKL_GRADING WRITING DATA 2024" \
  --output-dir data/selected_prompt_jsons_100 \
  --samples-per-prompt 100 \
  --seed 42
```

- `--samples-per-prompt`: 주제별 추출 개수. 기본값은 `100`.
- `--seed`: 재현 가능한 랜덤 선택을 위한 시드. 기본값은 `42`.
- `--output-dir`: 정제 JSON을 저장할 디렉토리.

## 출력 필드

각 JSON 파일은 다음 필드를 포함한다.

```json
[
  {
    "essay_id": "GWGR2400103100.1",
    "source_id": "국립국어원 글쓰기 채점 말뭉치 GWGR2400103100",
    "prompt_id": "Q9",
    "prompt_text": "...",
    "essay_text": "...",
    "meta": {
      "written_length": 1168,
      "paragraph_num": 3,
      "sentence_num_original": 19,
      "sentence_num_recounted": 19
    },
    "label_5scale_evaluator1": {"con": 3.0, "org": 3.0, "exp": 4.5},
    "label_5scale_evaluator2": {"con": 2.6, "org": 2.5, "exp": 3.0},
    "label_5scale_average": {"con": 2.8, "org": 2.75, "exp": 3.75}
  }
]
```
