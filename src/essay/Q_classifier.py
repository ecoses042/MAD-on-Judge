import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_prompt_samples_to_single_dir(
    input_dir: str | Path,
    output_dir: str | Path,
    prompt_ids: List[str] = None,
    max_count_per_prompt: int = 50,
) -> None:
    """
    input_dir 아래의 모든 json 파일을 재귀적으로 읽어서,
    각 파일 안의 essay 객체들 중 prompt_id가 Q4~Q9인 항목을 찾는다.

    각 prompt_id마다 최대 max_count_per_prompt개까지 뽑아서,
    객체 1개를 [객체] 형태의 단일 JSON 파일로 output_dir에 저장한다.

    예:
      output_dir/
        Q4_001_GWGR2400111090.1.json
        Q4_002_xxxxx.json
        Q6_001_GWGR2400111090.1.json
    """
    if prompt_ids is None:
        prompt_ids = ["Q4", "Q5", "Q6", "Q7", "Q8", "Q9"]

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {pid: 0 for pid in prompt_ids}

    json_files = sorted(input_dir.rglob("*.json"))

    for json_file in json_files:
        try:
            data = load_json(json_file)
        except Exception as e:
            print(f"[오류] 파일 읽기 실패: {json_file} -> {e}")
            continue

        if not isinstance(data, list):
            print(f"[경고] 리스트 형식이 아닌 파일 건너뜀: {json_file}")
            continue

        for item in data:
            if not isinstance(item, dict):
                continue

            prompt_id = str(item.get("prompt_id", "")).strip()
            if prompt_id not in prompt_ids:
                continue

            if counts[prompt_id] >= max_count_per_prompt:
                continue

            essay_id = str(item.get("essay_id", f"no_essay_id_{counts[prompt_id]+1}")).strip()
            safe_essay_id = essay_id.replace("/", "_").replace("\\", "_").replace(":", "_")

            counts[prompt_id] += 1
            output_name = f"{prompt_id}_{counts[prompt_id]:03d}_{safe_essay_id}.json"
            output_path = output_dir / output_name

            # 원본 객체 형식을 유지하기 위해 [item] 형태로 저장
            save_json([item], output_path)

            print(f"[저장] {output_name}")

        if all(counts[pid] >= max_count_per_prompt for pid in prompt_ids):
            break

    print("\n=== 완료 ===")
    for pid in prompt_ids:
        print(f"{pid}: {counts[pid]}개 저장")


if __name__ == "__main__":
    extract_prompt_samples_to_single_dir(
        input_dir="processed_examples_from_folder",
        output_dir="selected_prompt_jsons",
        prompt_ids=["Q4", "Q5", "Q6", "Q7", "Q8", "Q9"],
        max_count_per_prompt=50,
    )