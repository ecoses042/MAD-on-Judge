from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional
import shutil


# 영역별 최대점
MAX_SCORES = {
    "con": 25.0,  # 5개 항목 * 5점
    "org": 10.0,   # 2개 항목 * 5점
    "exp": 10.0,   # 2개 항목 * 5점
}


def safe_mean(values: List[float]) -> Optional[float]:
    """비어 있지 않은 숫자 리스트의 평균을 반환한다."""
    if not values:
        return None
    return float(mean(values))


def join_paragraphs(paragraphs: List[Dict[str, Any]]) -> str:
    """paragraph.form을 순서대로 이어 붙여 완성된 essay_text를 만든다."""
    texts: List[str] = []
    for para in paragraphs:
        text = str(para.get("form", "")).strip()
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def count_sentences(paragraphs: List[Dict[str, Any]]) -> int:
    """sentence 배열 기준으로 실제 문장 수를 재계산한다."""
    total = 0
    for para in paragraphs:
        sentences = para.get("sentence", [])
        if isinstance(sentences, list):
            total += len(sentences)
    return total


def parse_dimension_scores(evaluation_data: Dict[str, Any], dimension: str, evaluator: Optional[int] = None) -> Dict[str, Any]:
    """
    영역별(con/org/exp) raw 점수를 읽어 점수를 반환한다.
    evaluator: None이면 모든 평가자 점수, 1또는 2면 해당 평가자만
    반환 예시 (evaluator=None):
    {
        "raw": [14.0, 12.0],
        "avg": 13.0
    }
    반환 예시 (evaluator=1):
    {
        "raw": 14.0,
        "avg": 14.0
    }
    """
    key = f"eva_score_{dimension}"
    dim_block = evaluation_data.get(key, {})

    if evaluator is None:
        # 모든 평가자의 점수를 반환 (기존 동작)
        raw_scores: List[float] = []
        for k, v in dim_block.items():
            if k.endswith(f"score_total_{dimension}") and isinstance(v, (int, float)):
                raw_scores.append(float(v))
        return {
            "raw": raw_scores,
            "avg": safe_mean(raw_scores),
        }
    else:
        # 특정 평가자의 점수만 반환
        search_key = f"evaluator{evaluator}_score_total_{dimension}"
        for k, v in dim_block.items():
            if k == search_key and isinstance(v, (int, float)):
                score = float(v)
                return {
                    "raw": score,
                    "avg": score,
                }
        return {
            "raw": None,
            "avg": None,
        }


def parse_total_scores(evaluation_data: Dict[str, Any], evaluator: Optional[int] = None) -> Dict[str, Any]:
    """
    평가자 총점을 읽어 총점을 반환한다.
    evaluator: None이면 모든 평가자 점수, 1또는 2면 해당 평가자만
    반환 예시 (evaluator=None):
    {
        "raw": [27.0, 26.0],
        "avg": 26.5
    }
    반환 예시 (evaluator=1):
    {
        "raw": 27.0,
        "avg": 27.0
    }
    """
    if evaluator is None:
        # 모든 평가자의 점수를 반환 (기존 동작)
        raw_scores: List[float] = []
        for key in ["evaluator1_total_score", "evaluator2_total_score"]:
            value = evaluation_data.get(key)
            if isinstance(value, (int, float)):
                raw_scores.append(float(value))
        return {
            "raw": raw_scores,
            "avg": safe_mean(raw_scores),
        }
    else:
        # 특정 평가자의 점수만 반환
        key = f"evaluator{evaluator}_total_score"
        value = evaluation_data.get(key)
        if isinstance(value, (int, float)):
            score = float(value)
            return {
                "raw": score,
                "avg": score,
            }
        return {
            "raw": None,
            "avg": None,
        }


def extract_prompt_text(doc_metadata: Dict[str, Any]) -> str:
    """프롬프트 본문을 안전하게 추출한다."""
    prompt = doc_metadata.get("prompt", {})
    return str(prompt.get("prompt_con", "")).strip()


def extract_prompt_id(doc_metadata: Dict[str, Any]) -> str:
    """프롬프트 번호를 안전하게 추출한다."""
    prompt = doc_metadata.get("prompt", {})
    return str(prompt.get("prompt_num", "")).strip()


def normalize_to_5_scale(score: Optional[float], max_score: float) -> Optional[float]:
    """
    raw 평균 점수를 5점 척도로 변환한다.
    초기 실험에서는 소수점을 그대로 유지한다.
    """
    if score is None or max_score == 0:
        return None
    return (score / max_score) * 5.0


def normalize_dimension_scores(raw_mean_scores: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """
    영역별 평균 raw 점수를 5점 척도로 변환한다.
    """
    return {
        "con": normalize_to_5_scale(raw_mean_scores.get("con"), MAX_SCORES["con"]),
        "org": normalize_to_5_scale(raw_mean_scores.get("org"), MAX_SCORES["org"]),
        "exp": normalize_to_5_scale(raw_mean_scores.get("exp"), MAX_SCORES["exp"]),
    }


def transform_document(doc: Dict[str, Any], root_metadata: Dict[str, Any], evaluator: Optional[int] = None) -> Dict[str, Any]:
    """
    원본 document 1개를 대회용 평탄화 샘플 1개로 변환한다.
    evaluator: None이면 평균, 1또는 2면 해당 평가자 기준
    """
    doc_id = str(doc.get("id", "")).strip()
    doc_metadata = doc.get("metadata", {})
    paragraphs = doc.get("paragraph", [])
    evaluation = doc.get("evaluation", {})
    evaluation_data = evaluation.get("evaluation_data", {})

    essay_text = join_paragraphs(paragraphs)
    sentence_num_recount = count_sentences(paragraphs)

    written_stat = doc_metadata.get("written_stat", {})

    con_scores = parse_dimension_scores(evaluation_data, "con", evaluator)
    org_scores = parse_dimension_scores(evaluation_data, "org", evaluator)
    exp_scores = parse_dimension_scores(evaluation_data, "exp", evaluator)
    total_scores = parse_total_scores(evaluation_data, evaluator)

    label_raw_mean = {
        "con": con_scores["avg"],
        "org": org_scores["avg"],
        "exp": exp_scores["avg"],
        "total": total_scores["avg"],
    }

    label_5scale = normalize_dimension_scores(label_raw_mean)

    transformed = {
        "essay_id": doc_id,
        "source_id": root_metadata.get("title", ""),
        "prompt_id": extract_prompt_id(doc_metadata),
        "prompt_text": extract_prompt_text(doc_metadata),
        "essay_text": essay_text,
        "meta": {
            "written_length": written_stat.get("written_length"),
            "paragraph_num": written_stat.get("paragraph_num"),
            "sentence_num_original": written_stat.get("sentence_num"),
            "sentence_num_recounted": sentence_num_recount,
        },
        "label_raw_mean": label_raw_mean,
        "label_5scale": label_5scale,
    }

    return transformed


def transform_corpus(data: Dict[str, Any], evaluator: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    원본 JSON 전체를 받아 document 단위로 평탄화된 샘플 리스트를 반환한다.
    evaluator: None이면 평균, 1또는 2면 해당 평가자 기준
    """
    root_metadata = data.get("metadata", {})
    documents = data.get("document", [])

    results: List[Dict[str, Any]] = []
    for doc in documents:
        results.append(transform_document(doc, root_metadata, evaluator))

    return results


def transform_corpus_all_labels(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    원본 JSON을 받아 각 document마다 3가지 버전(평가자1, 평가자2, 평균)의 
    label_5scale을 모두 계산해서 하나의 document에 담아 반환한다.
    결과: 1개의 document = 1개의 항목
    """
    root_metadata = data.get("metadata", {})
    documents = data.get("document", [])

    results: List[Dict[str, Any]] = []
    
    for doc in documents:
        doc_id = str(doc.get("id", "")).strip()
        doc_metadata = doc.get("metadata", {})
        paragraphs = doc.get("paragraph", [])
        evaluation = doc.get("evaluation", {})
        evaluation_data = evaluation.get("evaluation_data", {})

        essay_text = join_paragraphs(paragraphs)
        sentence_num_recount = count_sentences(paragraphs)
        written_stat = doc_metadata.get("written_stat", {})

        # 3가지 버전 각각 계산
        label_5scales = {}
        for evaluator_type, label_key in [(1, "evaluator1"), (2, "evaluator2"), (None, "average")]:
            con_scores = parse_dimension_scores(evaluation_data, "con", evaluator_type)
            org_scores = parse_dimension_scores(evaluation_data, "org", evaluator_type)
            exp_scores = parse_dimension_scores(evaluation_data, "exp", evaluator_type)
            total_scores = parse_total_scores(evaluation_data, evaluator_type)

            label_raw_mean = {
                "con": con_scores["avg"],
                "org": org_scores["avg"],
                "exp": exp_scores["avg"],
                "total": total_scores["avg"],
            }
            
            label_5scale = normalize_dimension_scores(label_raw_mean)
            label_5scales[f"label_5scale_{label_key}"] = label_5scale

        # 하나의 document에 3가지 label_5scale 포함
        transformed = {
            "essay_id": doc_id,
            "source_id": root_metadata.get("title", ""),
            "prompt_id": extract_prompt_id(doc_metadata),
            "prompt_text": extract_prompt_text(doc_metadata),
            "essay_text": essay_text,
            "meta": {
                "written_length": written_stat.get("written_length"),
                "paragraph_num": written_stat.get("paragraph_num"),
                "sentence_num_original": written_stat.get("sentence_num"),
                "sentence_num_recounted": sentence_num_recount,
            },
        }
        
        # 3가지 label_5scale 추가
        transformed.update(label_5scales)
        
        results.append(transformed)

    return results


def load_json(path: str | Path) -> Dict[str, Any]:
    # UTF-8 BOM이 포함된 파일도 안전하게 읽도록 utf-8-sig 사용
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def get_prompt_id_from_file(file_path: Path) -> str:
    """JSON 파일에서 첫 번째 document의 prompt_id를 추출한다."""
    data = load_json(file_path)
    documents = data.get("document", [])
    if not documents:
        return ""
    doc_metadata = documents[0].get("metadata", {})
    return extract_prompt_id(doc_metadata)


def process_file(input_path: str | Path, output_path: str | Path, evaluator: Optional[int] = None) -> None:
    """
    단일 원본 JSON 파일을 읽어서 변환 결과를 저장한다.
    evaluator: None이면 평균, 1또는 2면 해당 평가자 기준
    """
    data = load_json(input_path)
    transformed = transform_corpus(data, evaluator)
    save_json(transformed, output_path)


def process_directory(input_dir: str | Path, output_dir: str | Path, evaluator: Optional[int] = None) -> None:
    """
    폴더 내 모든 JSON 파일을 일괄 처리한다.
    각 입력 파일마다 동일한 파일명으로 결과를 저장한다.
    evaluator: None이면 평균, 1또는 2면 해당 평가자 기준
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in input_dir.glob("*.json"):
        output_file = output_dir / input_file.name
        process_file(input_file, output_file, evaluator)


def process_directory_to_single_file(input_dir: str | Path, output_file: str | Path, evaluator: Optional[int] = None) -> None:
    """
    폴더 내 모든 JSON 파일을 읽어 하나의 리스트로 합친 뒤 단일 JSON 파일로 저장한다.
    (processed_example 형식: transform_corpus_all_labels 사용)
    evaluator: None이면 평균 + evaluator1/2 동시 포함. 1 또는 2 선택 시 단일 evaluator 기준.
    """
    input_dir = Path(input_dir)
    output_file = Path(output_file)

    all_results: List[Dict[str, Any]] = []
    for input_file in sorted(input_dir.glob("*.json")):
        data = load_json(input_file)
        if evaluator is None:
            transformed = transform_corpus_all_labels(data)
        else:
            transformed = transform_corpus(data, evaluator)
        all_results.extend(transformed)

    save_json(all_results, output_file)


def process_directory_all_labels(input_dir: str | Path, output_dir: str | Path) -> None:
    """
    폴더 내 모든 JSON 파일을 읽어 파일별로 convert 결과를 output_dir에 저장.
    transform_corpus_all_labels를 적용하여 모든 평가자 버전을 포함한다.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in input_dir.glob("*.json"):
        data = load_json(input_file)
        transformed = transform_corpus_all_labels(data)
        output_file = output_dir / input_file.name
        save_json(transformed, output_file)


if __name__ == "__main__":
    # 3가지 버전(평가자1, 평가자2, 평균) 모두를 하나의 JSON 파일에 저장

    # 현재 스크립트(src) 위치를 기준으로 작업 디렉터리 설정
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # 원본 데이터 폴더 이름 (공백/NBSP 변환을 고려하여 탐색)
    target_name = "NIKL_GRADING WRITING DATA 2024"

    def normalize(name: str) -> str:
        return name.replace("\u00A0", " ").strip().lower()

    def find_matching_dir(base: Path, desired: str) -> Optional[Path]:
        for p in base.iterdir():
            if p.is_dir() and normalize(p.name) == normalize(desired):
                return p
        return None

    root_data_dir = find_matching_dir(project_root, target_name)
    if root_data_dir is None:
        raise FileNotFoundError(
            f"Cannot find root data directory matching: {target_name}.\n"
            f"Available directories: {[d.name for d in project_root.iterdir() if d.is_dir()]}"
        )

    input_folder = root_data_dir
    output_dir = project_root / "processed_examples_from_folder"

    print('project_root:', project_root)
    print('input_folder:', input_folder)
    print('input_folder exists:', input_folder.exists())
    print('output_dir:', output_dir)

    # JSON 파일들을 prompt_id로 그룹화
    files_by_prompt = {}
    for json_file in input_folder.glob("*.json"):
        prompt_id = get_prompt_id_from_file(json_file)
        if prompt_id not in files_by_prompt:
            files_by_prompt[prompt_id] = []
        files_by_prompt[prompt_id].append(json_file)

    # q4부터 q9까지 각 prompt_id에 대해 50개씩 모아서 output_dir에 넣기
    for prompt_id in ['q4', 'q5', 'q6', 'q7', 'q8', 'q9']:
        if prompt_id not in files_by_prompt:
            print(f"No files found for prompt_id: {prompt_id}")
            continue
        files = files_by_prompt[prompt_id][:50]  # 처음 50개
        prompt_output_dir = output_dir / prompt_id
        prompt_output_dir.mkdir(parents=True, exist_ok=True)
        for file_path in files:
            shutil.copy(file_path, prompt_output_dir / file_path.name)
        print(f"Copied {len(files)} files for prompt_id {prompt_id} to {prompt_output_dir}")
    # 개별 처리 예시 (필요시)
    # process_file("raw_example.json", "processed_example_evaluator1.json", evaluator=1)
    # process_file("raw_example.json", "processed_example_evaluator2.json", evaluator=2)
    # process_file("raw_example.json", "processed_example_average.json", evaluator=None)
    
    # 폴더 일괄 처리 예시
    # process_directory("raw_jsons", "processed_jsons_evaluator1", evaluator=1)
    # process_directory("raw_jsons", "processed_jsons_evaluator2", evaluator=2)
    # process_directory("raw_jsons", "processed_jsons_average", evaluator=None)