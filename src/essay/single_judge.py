import argparse
import json
import os
import sys
import time
from pathlib import Path
from openai import OpenAI, RateLimitError
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from env_utils import load_project_env

load_project_env(__file__)

API_KEY = os.environ.get("OPENAI_API_KEY", "")
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

# LM Studio 모델 매핑: 인자 -> 실제 모델 ID
LM_STUDIO_MODELS = {
    "gemma": "google/gemma-3-4b",
    "qwen": "qwen/qwen3.5-9b",
}

INPUT_DIRS = ["inference_results/gemma", "inference_results/gpt", "inference_results/qwen", "inference_results/llama"]
ESSAY_MODELS = ["gemma", "gpt", "qwen", "llama"]
ESSAY_DATA_DIR = "data/selected_prompt_jsons"


def make_client(judge_model: str):
    """judge_model에 따라 적절한 OpenAI 클라이언트와 모델명을 반환한다."""
    if judge_model in LM_STUDIO_MODELS:
        client = OpenAI(api_key="lm-studio", base_url=LM_STUDIO_BASE_URL)
        model_name = LM_STUDIO_MODELS[judge_model]
        return client, model_name
    # else:
    #     # 기본: GPT (gpt-4o-mini)
    #     client = OpenAI(api_key=API_KEY)
    #     return client, "gpt-4o-mini"


def build_essay_index(essay_data_dir: Path) -> dict:
    """
    ESSAY_DATA_DIR 안의 모든 JSON 파일을 읽어
    { essay_id -> {"prompt_text": ..., "essay_text": ...} } 인덱스를 만든다.
    파일 하나에 여러 에세이가 있어도 모두 수집한다.
    """
    index = {}
    for json_file in essay_data_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue
            for item in data:
                essay_id = item.get("essay_id")
                if not essay_id:
                    continue
                index[essay_id] = {
                    "prompt_text": item.get("prompt_text", "").strip(),
                    "essay_text": item.get("essay_text", "").strip(),
                }
        except Exception as e:
            print(f"[WARNING] 인덱스 구축 실패: {json_file}, {e}")
    print(f"[INFO] 에세이 인덱스 구축 완료: {len(index)}개")
    return index

def compute_overall(result: dict) -> float | None:
    """
    4개 항목의 평균으로 overall 계산.
    치명적 약점이 있으면 상한 제한.
    """
    keys = ["domain_match", "score_rationale_consistency", "specificity", "groundedness"]
    scores = [result.get(k) for k in keys]

    if any(not isinstance(s, (int, float)) for s in scores):
        return None

    avg = round(sum(scores) / 4, 1)
    min_score = min(scores)

    # 치명적 결함 상한 제한
    if result.get("domain_match", 5) <= 2:
        avg = min(avg, 2.5)
    if result.get("groundedness", 5) <= 2:
        avg = min(avg, 2.5)

    # 최저점 기반 상한
    if min_score == 1:
        avg = min(avg, 2.0)
    elif min_score == 2:
        avg = min(avg, 3.0)

    return round(avg, 1)


def extract_json_object(raw_text: str) -> dict:
    """
    모델 응답에서 첫 번째 JSON 객체만 추출한다.
    - ```json ... ``` 코드블록
    - JSON 뒤에 붙는 설명/마크다운
    - JSON 앞에 붙는 짧은 안내문
    같은 흔한 실패 유형을 복구하는 용도다.
    """
    if not raw_text:
        raise json.JSONDecodeError("Empty response", raw_text, 0)

    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object start found", cleaned, 0)

    depth = 0
    in_string = False
    escape = False
    end = None

    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]

        if escape:
            escape = False
            continue

        if ch == "\\" and in_string:
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break

    if end is None:
        raise json.JSONDecodeError("No complete JSON object found", cleaned, start)

    return json.loads(cleaned[start:end])


def judge_essay(essay_text: str, prompt_text: str, target_domain: str,
                predicted_score: float, rationale: str,
                client: OpenAI = None, model_name: str = "gpt-4o-mini") -> dict | None:

    system_prompt = """
[역할]
당신은 한국어 에세이 채점 결과의 타당성을 검토하는 매우 엄격하고 일관된 심사자이다.
당신의 목표는 "점수와 근거의 품질 차이"를 실제로 구분해 내는 것이다.
애매하면 높은 점수를 주지 말고, 근거가 부족하면 분명히 감점하라.

[평가 대상]
모델이 특정 영역(target_domain)에 대해 제시한:
1) predicted_score
2) rationale
이 두 가지가 essay_text에 비추어 얼마나 타당한지 평가한다.

[중요 원칙]
- 1~5 전 구간을 적극적으로 사용하라.
- 기본 점수는 3점이다. 명확한 강점이 입증될 때만 4~5점을 준다.
- "그럴듯해 보인다"는 이유만으로 4점을 주지 마라.
- 에세이의 구체적 표현, 문장, 문단 구조, 실제 논지와 연결되지 않으면 specificity/groundedness는 높게 줄 수 없다.
- generic한 총평, 상투적 표현, 템플릿형 설명은 낮게 평가하라.
- essay_text에 없는 내용을 암시하거나 만들어내면 groundedness는 반드시 1~2점이다.
- target_domain과 다른 영역 기준을 섞으면 domain_match는 반드시 1~2점이다.
- predicted_score가 높은데 rationale이 약하거나 generic하면 score_rationale_consistency는 낮게 줘야 한다.
- predicted_score가 낮은데 rationale이 실제 결함을 충분히 입증하지 못하면 score_rationale_consistency는 낮게 줘야 한다.
- "대체로 괜찮다", "전반적으로 무난하다" 같은 표현만으로는 높은 specificity를 줄 수 없다.

[평가 항목]
1. domain_match
- rationale이 target_domain의 평가 기준에 맞는 근거를 제시하는가

2. score_rationale_consistency
- predicted_score와 rationale의 내용이 서로 잘 맞는가

3. specificity
- rationale이 구체적인가
- 실제 글의 특정 문장, 표현, 논지, 문단 전개, 오류 양상을 짚는가

4. groundedness
- rationale이 실제 essay_text에 근거하는가
- 에세이에 없는 내용을 만들어내지 않는가

[target_domain별 판단 기준]
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

[강제 감점 규칙]
다음 중 하나라도 해당하면 관련 항목은 2점 이하를 적극 검토하라.
- target_domain과 무관한 기준으로 설명함
- essay_text에 없는 내용을 근거처럼 제시함
- predicted_score를 정당화할 핵심 증거가 없음
- 지나치게 일반적이고 템플릿 같은 설명만 함
- 실제 글의 장점/결함을 확인 가능한 수준으로 짚지 못함

[점수 기준]
5점:
- 매우 강한 타당성
- 구체적이고 정확하며 essay_text와 긴밀히 연결됨
- 해당 영역 기준을 정확히 적용함
- predicted_score를 설득력 있게 정당화함

4점:
- 전반적으로 타당함
- 다소 아쉬움은 있으나 충분히 근거 있고 구체적임
- essay_text와의 연결이 분명함

3점:
- 기본적인 타당성은 있음
- 그러나 구체성, 명확성, 근거 연결 중 하나 이상이 부족함
- generic하거나 부분적으로만 설득력 있음

2점:
- 문제점이 뚜렷함
- 영역 혼동, 근거 부족, 일반론적 설명, essay와의 연결 부족이 보임
- predicted_score를 신뢰하기 어려움

1점:
- 거의 타당하지 않음
- 명백한 영역 혼동, 환각, 근거 부재, 점수-설명 모순이 있음

[항목별 세부 판정 기준]

[domain_match]
5: 해당 영역 기준만 정확히 사용
4: 대부분 정확하나 약간의 경계 혼합
3: 대체로 맞지만 일부 다른 영역 요소가 섞임
2: 다른 영역 기준이 꽤 섞여 있음
1: 거의 다른 영역 기준으로 설명

[score_rationale_consistency]
5: 점수와 설명이 매우 잘 부합
4: 대체로 부합
3: 어느 정도는 맞지만 애매함
2: 점수 대비 근거가 약하거나 불충분
1: 점수와 설명이 명확히 모순

[specificity]
5: 실제 글의 특정 요소를 분명히 짚음
4: 비교적 구체적임
3: 최소한의 근거는 있으나 다소 일반적임
2: 상당히 추상적이고 뭉뚱그린 설명
1: 거의 템플릿형 총평 수준

[groundedness]
5: 실제 essay_text에 명확히 근거
4: 대체로 근거
3: 부분적으로만 근거, 일부 추정 존재
2: essay와의 연결이 약하고 확인 어려운 판단이 많음
1: essay에 없는 내용을 말하거나 근거성이 매우 약함

[출력 규칙]
- 설명 문장은 출력하지 말고 JSON 객체 하나만 출력하라.
- 코드블록 마크다운(```)을 사용하지 마라.
- 모든 점수는 반드시 1~5의 정수여야 한다.

[출력 형식]
{
  "domain_match": 1,
  "score_rationale_consistency": 1,
  "specificity": 1,
  "groundedness": 1
}

이제 아래 정보를 바탕으로 평가하라.
"""

    user_prompt = f"""
[target_domain]
{target_domain}

[predicted_score]
{predicted_score}

[rationale]
{rationale}

[essay_prompt]
{prompt_text}

[essay_text]
{essay_text}
"""

    # .format() 대신 f-string 사용 → 중괄호 충돌 없음

    raw_response = None

    # LM Studio 로컬 모델은 response_format을 지원하지 않을 수 있으므로 GPT만 사용
    use_json_format = model_name.startswith("gpt-")

    for attempt in range(3):
        try:
            create_kwargs = dict(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=512,
            )
            if use_json_format:
                create_kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**create_kwargs)

            raw_response = response.choices[0].message.content.strip()
            result = extract_json_object(raw_response)

            required_fields = [
                "domain_match",
                "score_rationale_consistency",
                "specificity",
                "groundedness",
            ]

            for field in required_fields:
                if field not in result:
                    print(f"[WARNING] 필드 누락: {field}")
                    return None

            # 점수값 타입 보정 및 범위 클램핑
            for key in required_fields:
                val = result.get(key)
                if not isinstance(val, int):
                    try:
                        val = int(val)
                    except Exception:
                        print(f"[WARNING] 점수 타입 오류: {key}={result.get(key)}")
                        return None
                result[key] = max(1, min(5, val))

            # overall은 모델에게 받지 않고 직접 계산
            result["overall_judge"] = compute_overall(result)
            return result

        except RateLimitError:
            wait = 10 * (attempt + 1)
            print(f"[Rate Limit] {attempt + 1}번째 재시도, {wait}초 대기...")
            time.sleep(wait)
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON 파싱 오류: {e} | raw: {raw_response}")
            return None
        except Exception as e:
            print(f"[ERROR] API 호출 오류: {e}")
            return None

    print("[ERROR] 3회 재시도 모두 실패")
    return None


def process_folder(input_folder: str, output_folder: str, essay_index: dict,
                   client: OpenAI = None, model_name: str = "gpt-4o-mini"):
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    for json_file in sorted(input_dir.glob("*.json")):

        output_file = output_dir / json_file.name
        if output_file.exists():
            print(f"[SKIP] 이미 처리됨: {json_file.name}")
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARNING] JSON 읽기 실패 {json_file}: {e}")
            continue

        if not isinstance(data, list) or not data:
            print(f"[WARNING] 예상치 못한 JSON 구조: {json_file}")
            continue

        judged_data = []

        for item in data:
            essay_id = item.get("essay_id")
            if not essay_id:
                print(f"[WARNING] essay_id 없음, 건너뜀")
                continue

            # essay_id 기반으로 essay 데이터 조회
            essay_data = essay_index.get(essay_id)
            if not essay_data:
                print(f"[WARNING] 인덱스에서 essay_id를 찾을 수 없음: {essay_id}")
                continue

            prompt_text = essay_data["prompt_text"]
            essay_text = essay_data["essay_text"]

            if not essay_text:
                print(f"[WARNING] essay_text가 비어 있음: {essay_id}")
                continue

            prediction = item.get("prediction", {})
            prediction_normalized = {k.lower(): v for k, v in prediction.items()}

            judged_item = {
                "essay_id": essay_id,
                "source_id": item.get("source_id"),
                "prompt_id": item.get("prompt_id"),
                "status": item.get("status"),
                "gold": item.get("gold"),
                "prediction": {},
                "judge": {}
            }

            for domain in ["content", "organization", "expression"]:
                if domain not in prediction_normalized:
                    continue

                domain_data = prediction_normalized[domain]
                score = domain_data.get("score")
                rationale = domain_data.get("rationale")

                if score is None or rationale is None:
                    print(f"[WARNING] score/rationale 누락: {essay_id}, {domain}")
                    continue

                judge_result = judge_essay(
                    essay_text=essay_text,
                    prompt_text=prompt_text,
                    target_domain=domain.capitalize(),
                    predicted_score=score,
                    rationale=rationale,
                    client=client,
                    model_name=model_name,
                )

                if judge_result:
                    judged_item["prediction"][domain] = domain_data
                    judged_item["judge"][domain] = judge_result
                else:
                    print(f"[ERROR] Judge 실패: {essay_id}, {domain}")

            judged_data.append(judged_item)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(judged_data, f, ensure_ascii=False, indent=2)

        print(f"처리 완료: {json_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="단일 LLM Judge로 에세이 채점 결과 평가")
    parser.add_argument(
        "--judge-model",
        default="gemma",
        choices=["gemma", "qwen"],
        help="Judge로 사용할 모델: gpt(기본, GPT-4o-mini), gemma(LM Studio google/gemma-3-4b), qwen(LM Studio qwen/qwen3.5-9b)",
    )
    args = parser.parse_args()

    client, model_name = make_client(args.judge_model)
    print(f"[INFO] Judge 모델: {model_name}")

    if args.judge_model == "gpt" and not API_KEY:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

    # 에세이 인덱스를 한 번만 구축
    essay_index = build_essay_index(Path(ESSAY_DATA_DIR))

    if not essay_index:
        raise RuntimeError(f"에세이 인덱스가 비어 있습니다. {ESSAY_DATA_DIR} 경로를 확인하세요.")

    output_dirs = [f"judge_results/single_judge/{args.judge_model}/{essay_model}" for essay_model in ESSAY_MODELS]

    for input_dir, output_dir in zip(INPUT_DIRS, output_dirs):
        input_folder = Path(input_dir)

        if not input_folder.exists():
            print(f"[WARNING] 입력 폴더 없음, 건너뜀: {input_folder}")
            continue

        print(f"\n=== 처리 시작: {input_dir} → {output_dir} ===")
        process_folder(str(input_folder), output_dir, essay_index,
                       client=client, model_name=model_name)
