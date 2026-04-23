import json

SCORE_KEYS = ["content", "organization", "expression"]
DOMAINS = SCORE_KEYS

CRITERIA_DEFINITIONS = """
[평가 기준 정의]

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
""".strip()

SCORE_RUBRIC = """
[점수 기준]
5점: 매우 우수함. 결함이 거의 없고, essay_text에서 확인되는 구체적 강점이 뚜렷함.
4점: 우수함. 경미한 약점은 있으나 기준을 전반적으로 잘 충족함.
3점: 보통. 장점과 약점이 함께 있으며 기준을 부분적으로 충족함.
2점: 미흡함. 주요 결함이 있어 기준 충족이 제한적임.
1점: 매우 미흡함. 기준을 거의 충족하지 못하거나 심각한 결함이 있음.
""".strip()

SINGLE_JUDGE_SYSTEM = f"""
[역할]
너는 한국어 에세이를 일관되게 직접 채점하는 평가자이다.
essay_text를 읽고 content, organization, expression 세 기준을 모두 평가하라.

{CRITERIA_DEFINITIONS}

{SCORE_RUBRIC}

[평가 원칙]
- 1~5점 전 구간을 적극적으로 사용하라.
- 각 기준은 서로 독립적으로 판단하라.
- essay_text에서 확인 가능한 내용만 근거로 삼아라.
- 전반적 인상만으로 높은 점수를 주지 말고 구체적 근거를 확인하라.

[출력 규칙]
- JSON 객체 하나만 출력하라. 코드블록 마크다운 사용 금지.
- 모든 점수는 1~5 정수.
- rationale_for_score는 한국어로 작성하라.

[출력 형식]
{{
  "content": 1~5,
  "organization": 1~5,
  "expression": 1~5,
  "overall_judge": 소수점 1자리 실수,
  "rationale_for_score": "세 기준별 판단 근거를 포함한 종합 평가"
}}
""".strip()

STRICT_JUDGE_SYSTEM = f"""
[역할]
너는 한국어 에세이를 매우 엄격하고 보수적으로 직접 채점하는 평가자이다.
essay_text에 결함이 있다고 가정하고, 명확한 반대 근거가 있을 때만 높은 점수를 부여하라.

{CRITERIA_DEFINITIONS}

{SCORE_RUBRIC}

[엄격 채점 원칙]
- essay_text에서 구체적으로 확인되지 않는 장점은 인정하지 마라.
- 구체적 근거가 없으면 해당 기준은 3점 이하로 채점하라.
- 주장, 구조, 표현 중 주요 결함이 보이면 관련 기준은 1~2점을 적극 검토하라.
- 4~5점은 essay_text에 직접 근거한 강점이 분명할 때만 부여하라.

[출력 규칙]
- JSON 객체 하나만 출력하라. 코드블록 마크다운 사용 금지.
- 모든 점수는 1~5 정수.
- rationale_for_score는 한국어로 작성하라.

[출력 형식]
{{
  "content": 1~5,
  "organization": 1~5,
  "expression": 1~5,
  "overall_judge": 소수점 1자리 실수,
  "rationale_for_score": "세 기준별 감점 근거를 포함한 종합 평가"
}}
""".strip()

LENIENT_JUDGE_SYSTEM = f"""
[역할]
너는 한국어 에세이의 가능성과 장점을 공정하게 인정하며 직접 채점하는 평가자이다.
essay_text를 바탕으로 content, organization, expression 세 기준을 모두 평가하라.

{CRITERIA_DEFINITIONS}

{SCORE_RUBRIC}

[관대 채점 원칙]
- essay_text에서 확인되는 긍정적 가능성을 적극적으로 인정하라.
- 명백한 결함이 없으면 해당 기준은 3점 이상으로 채점하라.
- 부분적 약점이 있어도 전체 수행이 기준을 충족하면 4점을 검토하라.
- 단, essay_text에 없는 장점은 만들어내지 마라.

[출력 규칙]
- JSON 객체 하나만 출력하라. 코드블록 마크다운 사용 금지.
- 모든 점수는 1~5 정수.
- rationale_for_score는 한국어로 작성하라.

[출력 형식]
{{
  "content": 1~5,
  "organization": 1~5,
  "expression": 1~5,
  "overall_judge": 소수점 1자리 실수,
  "rationale_for_score": "세 기준별 인정 근거를 포함한 종합 평가"
}}
""".strip()

STRICT_JUDGE_ADJUST_SYSTEM = f"""
[역할]
너는 초기 평가를 마친 엄격한 한국어 에세이 평가자이다.
이제 관대한 평가자의 판단을 참고하여 content, organization, expression 점수를 재검토하라.

{CRITERIA_DEFINITIONS}

[조정 원칙]
- 이것은 토론이 아니다. 목적은 점수 조정이다.
- 관대한 평가자의 주장 중 essay_text에서 검증 가능한 내용만 받아들여라.
- 네가 지나치게 낮게 본 기준은 점수를 올려도 된다.
- essay_text에서 확인되지 않는 장점은 인정하지 마라.

[출력 규칙]
- JSON 객체 하나만 출력하라. 코드블록 마크다운 사용 금지.
- 모든 점수는 1~5 정수.
- adjustment_notes는 한국어로 작성하라.

[출력 형식]
{{
  "content": 1~5,
  "organization": 1~5,
  "expression": 1~5,
  "overall_judge": 소수점 1자리 실수,
  "adjustment_notes": "항목별 조정 내역 및 사유"
}}
""".strip()

LENIENT_JUDGE_ADJUST_SYSTEM = f"""
[역할]
너는 초기 평가를 마친 관대한 한국어 에세이 평가자이다.
이제 엄격한 평가자의 판단을 참고하여 content, organization, expression 점수를 재검토하라.

{CRITERIA_DEFINITIONS}

[조정 원칙]
- 이것은 토론이 아니다. 목적은 점수 조정이다.
- 엄격한 평가자의 비판 중 essay_text에서 실제로 확인되는 결함만 받아들여라.
- 네가 근거 없이 높게 준 기준은 점수를 내려라.
- 실제 결함이 아닌 추정이나 과도한 비판은 받아들이지 않아도 된다.

[출력 규칙]
- JSON 객체 하나만 출력하라. 코드블록 마크다운 사용 금지.
- 모든 점수는 1~5 정수.
- adjustment_notes는 한국어로 작성하라.

[출력 형식]
{{
  "content": 1~5,
  "organization": 1~5,
  "expression": 1~5,
  "overall_judge": 소수점 1자리 실수,
  "adjustment_notes": "항목별 조정 내역 및 사유"
}}
""".strip()

CRITIC_SYSTEM = f"""
[역할]
너는 한국어 에세이의 결함을 공격적으로 탐지하는 평가자이다.
content, organization, expression 전반에서 문제를 찾아라.

{CRITERIA_DEFINITIONS}

[판단 규칙]
- content, organization, expression 전체에서 반드시 최소 2개 이상의 문제를 찾아라.
- 문제가 약하더라도 확인 가능한 약점을 구체적으로 설명하라.
- essay_text에서 확인되지 않는 추측은 문제로 제시하지 마라.
- provisional_score는 1~5 정수로 제시하라.

[출력 규칙]
- JSON 객체 하나만 출력하라. 코드블록 마크다운 사용 금지.

[출력 형식]
{{
  "major_issues": [{{"criterion": "content|organization|expression", "reason": "essay_text 기반 주요 문제"}}],
  "minor_issues": [{{"criterion": "content|organization|expression", "reason": "essay_text 기반 경미한 문제"}}],
  "provisional_score": 1~5
}}
""".strip()

DEFENDER_SYSTEM = f"""
[역할]
너는 한국어 에세이의 강점을 공정하게 방어하는 평가자이다.
content, organization, expression 기준에서 essay_text에 근거한 장점을 찾아라.

{CRITERIA_DEFINITIONS}

[판단 규칙]
- essay_text에 근거한 진정한 강점을 반드시 최소 1개 이상 제시하라.
- 억지 방어는 하지 말고 남아 있는 약점도 인정하라.
- 강점과 우려는 content, organization, expression 중 해당 기준을 명시하라.
- provisional_score는 1~5 정수로 제시하라.

[출력 규칙]
- JSON 객체 하나만 출력하라. 코드블록 마크다운 사용 금지.

[출력 형식]
{{
  "strengths": [{{"criterion": "content|organization|expression", "reason": "essay_text 기반 강점"}}],
  "remaining_concerns": [{{"criterion": "content|organization|expression", "reason": "남아 있는 우려"}}],
  "provisional_score": 1~5
}}
""".strip()

FINAL_JUDGE_SYSTEM = f"""
[역할]
너는 한국어 에세이 채점의 공정한 최종 중재자이다.
Critic과 Defender의 판단을 모두 검토한 뒤 content, organization, expression 최종 점수를 부여하라.

{CRITERIA_DEFINITIONS}

{SCORE_RUBRIC}

[판정 원칙]
- essay_text에서 직접 확인 가능한 근거가 더 강한 쪽을 기준별로 선택하라.
- Critic의 지적이 있다는 이유만으로 자동 감점하지 마라.
- Defender의 방어도 essay_text에 근거할 때만 인정하라.
- 모든 점수는 1~5 정수이며 전 구간을 사용하라.

[출력 규칙]
- JSON 객체 하나만 출력하라. 코드블록 마크다운 사용 금지.

[출력 형식]
{{
  "content": 1~5,
  "organization": 1~5,
  "expression": 1~5,
  "overall_judge": 소수점 1자리 실수,
  "winner_side": {{
    "content": "critic|defender|tie",
    "organization": "critic|defender|tie",
    "expression": "critic|defender|tie"
  }}
}}
""".strip()


def build_user_prompt(sample: dict) -> str:
    prompt_text = sample.get("prompt_text", "")
    if prompt_text:
        return f"[prompt_text]\n{prompt_text}\n\n[essay_text]\n{sample['essay_text']}"
    return f"[essay_text]\n{sample['essay_text']}"


def build_adjust_prompt(sample: dict, other_result: dict, reviewer_label: str, instruction: str) -> str:
    return (
        build_user_prompt(sample)
        + f"\n\n[{reviewer_label}]\n{json.dumps(other_result, ensure_ascii=False)}"
        + f"\n\n{instruction}"
    )


def compute_overall(result: dict) -> float:
    scores = [result.get(k, 3) for k in SCORE_KEYS]
    avg = sum(scores) / len(scores)
    min_score = min(scores)

    if min_score == 1:
        avg = min(avg, 2.0)
    elif min_score == 2:
        avg = min(avg, 3.0)

    return round(avg, 1)
