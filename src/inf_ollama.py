import re
import json
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from openai import OpenAI

# =========================
# Config
# =========================
MODEL_NAMES = [
        "llama3:8b",
]

TEMPERATURE = 0.0
SLEEP_SECONDS = 0.2

OLLAMA_BASE_URL = "http://localhost:11434/v1"
API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
def try_repair_json(text: str) -> str:
    open_braces = text.count("{")
    close_braces = text.count("}")
    if close_braces < open_braces:
        text += "}" * (open_braces - close_braces)
    return text
# =========================
# I/O
# =========================
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# =========================
# Utils
# =========================
def extract_json_object(text: str) -> str:
    text = text.strip()

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start:end + 1]

    raise ValueError("응답에서 JSON 객체를 찾지 못했습니다.")

def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def validate_score(score: Any) -> Optional[float]:
    score = safe_float(score)
    if score is None:
        return None
    if score < 1.0:
        score = 1.0
    if score > 5.0:
        score = 5.0
    return round(score, 4)

def normalize_prediction(pred: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result = {}

    mapping = {
        "content": ["content", "con"],
        "organization": ["organization", "org"],
        "expression": ["expression", "exp"],
    }

    for std_key, aliases in mapping.items():
        found = None
        for alias in aliases:
            if alias in pred and isinstance(pred[alias], dict):
                found = pred[alias]
                break

        if found is None:
            result[std_key] = {"score": None, "rationale": None}
            continue

        score = validate_score(found.get("score"))
        rationale = found.get("rationale")
        if rationale is not None:
            rationale = str(rationale).strip()

        result[std_key] = {
            "score": score,
            "rationale": rationale
        }

    return result

# =========================
# Prompt
# =========================
def build_messages(prompt_text: str, essay_text: str) -> List[Dict[str, str]]:
    system_prompt = """
당신은 한국어 논증형 에세이를 평가하는 숙련된 채점자이다.

주어진 에세이를 다음 3개 영역에서 각각 평가하라.

평가 영역
1. content
- 글의 주장과 핵심 내용이 문제에 적절하게 대응하는가
- 근거가 충분하고 구체적인가
- 주장과 근거 사이의 논리적 연결이 타당한가

2. organization
- 서론, 본론, 결론의 구조가 드러나는가
- 문단 간 연결이 자연스러운가
- 논리 전개 순서가 일관적인가

3. expression
- 문장이 자연스럽고 이해하기 쉬운가
- 어휘 사용이 적절한가
- 맞춤법, 띄어쓰기, 문법, 주술 호응에 문제가 없는가

점수 기준
각 영역은 반드시 1, 2, 3, 4, 5 중 하나의 정수 점수만 부여하라.

[content]
5점: 질문에 매우 적절하게 답하며, 주장이 명확하고 근거가 충분하고 구체적이다.
4점: 질문에 적절하게 답하며, 주장이 비교적 분명하고 근거도 대체로 충분하다.
3점: 기본적인 주장과 근거는 있으나, 설명이 피상적이거나 일부 논리 연결이 약하다.
2점: 주장이나 근거가 불충분하고, 질문에 대한 대응이 약하거나 논리 전개가 자주 흔들린다.
1점: 질문에 제대로 대응하지 못하거나, 주장과 근거가 거의 없고 내용이 매우 부실하다.

[organization]
5점: 글 전체 구조가 매우 분명하며, 문단과 문단 사이의 연결이 자연스럽고 논리 흐름이 안정적이다.
4점: 전체 구조가 비교적 분명하고, 전개 흐름도 대체로 자연스럽다.
3점: 기본 구조는 있으나 문단 연결이나 전개 순서가 다소 어색하다.
2점: 구조가 불안정하고, 문단 배열이나 전개 흐름이 자주 끊긴다.
1점: 구조가 거의 없거나 매우 혼란스러워 글의 흐름을 따라가기 어렵다.

[expression]
5점: 문장이 매우 자연스럽고 정확하며, 어휘 사용과 문법이 우수하다.
4점: 전반적으로 자연스럽고 큰 표현상 문제는 없으나, 일부 어색한 표현이 있을 수 있다.
3점: 이해는 가능하나, 표현의 단조로움이나 문법적 어색함이 일부 보인다.
2점: 문장 표현이 자주 어색하거나 문법, 맞춤법, 주술 호응 문제가 눈에 띈다.
1점: 표현 오류가 심하여 의미 전달이 어렵거나 문장 완성도가 매우 낮다.

중요 규칙
- 반드시 에세이의 실제 내용에 근거하여 평가하라.
- 각 영역의 rationale에는 반드시 에세이의 구체적 근거를 포함하라.
- "전반적으로 좋다", "무난하다", "대체로 적절하다" 같은 일반론만 쓰지 마라.
- 최소 1개 이상의 구체적 근거를 반드시 언급하라.
- 다른 영역 기준을 섞어 쓰지 마라.
  - content에서는 주장, 근거, 논리적 타당성을 중심으로 써라.
  - organization에서는 구조, 문단 전개, 연결을 중심으로 써라.
  - expression에서는 문장, 어휘, 문법, 맞춤법을 중심으로 써라.
- evidence는 에세이의 특정 표현, 문장, 또는 문단 수준의 근거를 짧게 요약한 문자열 배열로 작성하라.
- 출력은 반드시 JSON 객체 하나만 출력하라.
- 코드블록 마크다운(```)은 사용하지 마라.
- score는 반드시 정수형으로 작성하라.

출력 형식
{
  "content": {
    "score": 1~5 정수,
    "rationale": "구체적인 평가 설명",
    "evidence": ["근거1", "근거2"]
  },
  "organization": {
    "score": 1~5 정수,
    "rationale": "구체적인 평가 설명",
    "evidence": ["근거1", "근거2"]
  },
  "expression": {
    "score": 1~5 정수,
    "rationale": "구체적인 평가 설명",
    "evidence": ["근거1", "근거2"]
  }
}

이제 아래 정보를 바탕으로 평가하라.

[문항]
{prompt_text}

[에세이]
{essay_text}
""".strip()

    user_prompt = f"""
[문항]
{prompt_text}

[에세이]
{essay_text}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

# =========================
# Model call
# =========================
def call_model(client: OpenAI, model_name: str, prompt_text: str, essay_text: str) -> Dict[str, Any]:
    messages = build_messages(prompt_text, essay_text)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=TEMPERATURE,
        )

        raw_text = response.choices[0].message.content.strip()

        try:
            json_text = extract_json_object(raw_text)
            json_text = try_repair_json(json_text)
            parsed = json.loads(json_text)
            normalized = normalize_prediction(parsed)

            return {
                "status": "ok",
                "raw_response": raw_text,
                "prediction": normalized
            }

        except Exception as e:
            return {
                "status": "parse_error",
                "error": str(e),
                "raw_response": raw_text,
                "prediction": {
                    "content": {"score": None, "rationale": None},
                    "organization": {"score": None, "rationale": None},
                    "expression": {"score": None, "rationale": None},
                }
            }

    except Exception as e:
        return {
            "status": "api_error",
            "error": str(e),
            "raw_response": None,
            "prediction": {
                "content": {"score": None, "rationale": None},
                "organization": {"score": None, "rationale": None},
                "expression": {"score": None, "rationale": None},
            }
        }

# =========================
# Main processing
# =========================
def process_single_dataset(client: OpenAI, input_path: str, output_path: str, dataset_label: str, model_name: str) -> None:
    print(f"\n{'='*60}")
    print(f"비{dataset_label} 처리 시작: {input_path}")
    print(f"모델: {model_name}")
    print(f"{'='*60}")

    try:
        data = load_json(input_path)
    except FileNotFoundError:
        print(f"[ERROR] 파일을 찾지 못했습니다: {input_path}")
        return
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 구문 오류: {e}")
        return

    if not isinstance(data, list):
        print(f"[ERROR] {input_path}은 list 형태여야 합니다.")
        return

    results = []
    error_count = 0
    success_count = 0

    raw_response_dir = Path(output_path).parent / "raw_responses"
    raw_response_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(data):
        essay_id = item.get("essay_id", f"sample_{idx}")
        source_id = item.get("source_id")
        prompt_id = item.get("prompt_id")
        prompt_text = item.get("prompt_text", "")
        essay_text = item.get("essay_text", "")

        gold_scores = {}
        for label_key in ["evaluator1", "evaluator2", "average"]:
            label_5scale = item.get(f"label_5scale_{label_key}", {})
            gold_scores[label_key] = {
                "content": validate_score(label_5scale.get("con")),
                "organization": validate_score(label_5scale.get("org")),
                "expression": validate_score(label_5scale.get("exp"))
            }

        if not essay_text.strip():
            results.append({
                "essay_id": essay_id,
                "source_id": source_id,
                "prompt_id": prompt_id,
                "model": model_name,
                "status": "missing_essay_text",
                "gold": gold_scores,
                "prediction": {
                    "content": {"score": None, "rationale": None},
                    "organization": {"score": None, "rationale": None},
                    "expression": {"score": None, "rationale": None}
                },
                "raw_response": None
            })
            error_count += 1
            continue

        result = call_model(client, model_name, prompt_text, essay_text)

        results.append({
            "essay_id": essay_id,
            "source_id": source_id,
            "prompt_id": prompt_id,
            "model": model_name,
            "status": result["status"],
            "gold": gold_scores,
            "prediction": result["prediction"],
            "raw_response": result["raw_response"],
            "error": result.get("error")
        })

        # raw_response를 별도 파일로 저장
        raw_resp = result.get("raw_response")
        if raw_resp is not None:
            out_raw_name = f"{essay_id}_raw_response.json" if essay_id else f"{idx+1:04d}_raw_response.json"
            raw_file_path = raw_response_dir / out_raw_name
            with open(raw_file_path, "w", encoding="utf-8") as rf:
                json.dump({"essay_id": essay_id, "raw_response": raw_resp}, rf, ensure_ascii=False, indent=2)

        if result["status"] == "ok":
            success_count += 1
        else:
            error_count += 1

        print(f"[{idx + 1}/{len(data)}] {essay_id} - Status: {result['status']}")
        time.sleep(SLEEP_SECONDS)

    save_json(results, output_path)
    print(f"\n[{dataset_label}] 처리 완료")
    print(f"  - 집계: {len(data)}개")
    print(f"  - 성공: {success_count}개")
    print(f"  - 실패: {error_count}개")
    print(f"  - 출력: {output_path}\n")

def process_directory_dataset(client: OpenAI, input_dir: str, output_dir: str, dataset_label: str, model_name: str) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"디렉토리 {dataset_label} 처리 시작: {input_path}")
    print(f"{'='*60}")

    if not input_path.exists():
        print(f"[ERROR] 디렉토리를 찾을 수 없습니다: {input_path}")
        return

    json_files = list(input_path.glob("*.json"))
    if not json_files:
        print(f"[ERROR] {input_path}에 JSON 파일이 없습니다.")
        return

    print(f"발견된 파일 수: {len(json_files)}")

    for file_idx, json_file in enumerate(sorted(json_files)):
        file_name = json_file.stem
        output_file = output_path / f"{file_name}_results.json"

        print(f"\n--- 파일 {file_idx + 1}/{len(json_files)}: {file_name} ---")
        process_single_dataset(
            client,
            str(json_file),
            str(output_file),
            f"{dataset_label}_{file_name}",
            model_name
        )

    print(f"\n[{dataset_label}] 디렉토리 전체 처리 완료")
    print(f"  - 처리된 파일: {len(json_files)}개")
    print(f"  - 출력 디렉토리: {output_path}\n")

def main():
    client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=API_KEY,
    )

    for model_name in MODEL_NAMES:
        safe_model_name = model_name.replace(":", "_").replace("/", "_")

        process_directory_dataset(
            client,
            input_dir="selected_prompt_jsons",
            output_dir=f"inference_results/{safe_model_name}",
            dataset_label=f"폴더데이터셋_{safe_model_name}",
            model_name=model_name
        )

if __name__ == "__main__":
    main()