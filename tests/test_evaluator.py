"""최종 보고서 evaluator 규칙층 — LLM 없이 실행한다."""
from copy import deepcopy

from agents import evaluator
from graph.state import init_state

WEB_REFS = [
    "https://market-a.example/kivi",
    "https://market-b.example/infinigen",
    "https://stake-a.example/kivi",
    "https://stake-b.example/infinigen",
    "https://domain-a.example/kivi",
    "https://domain-b.example/infinigen",
]


def _state():
    state = init_state({}, {})
    state["tech_summary"] = {
        tech: {
            "overview": "개요", "mechanism": "원리", "numbers": ["수치 [p.1]"],
            "limitations": ["한계 [p.2]"], "apply_conditions": ["조건 [p.3]"], "evidence": [],
        }
        for tech in ("KIVI", "InfiniGen")
    }
    for index, key in enumerate(("market_eval", "stakeholder_eval", "domain_eval")):
        state[key] = {}
        for tech_index, tech in enumerate(("KIVI", "InfiniGen")):
            ref = WEB_REFS[index * 2 + tech_index]
            value = {
                "grade": "중", "rationale": "근거 [웹 " + ref + "]",
                "positives": ["긍정 [웹 " + ref + "]"],
                "negatives": ["반대 [웹 " + ref + "]"],
                "evidence": [{"claim": "근거", "tag": "웹", "ref": ref, "page": None}],
                "confidence": 0.8,
            }
            if key == "domain_eval":
                value |= {"verdict": "조건부", "axes": {"recall": "유지", "latency": "조건", "memory": "절감"}}
            state[key][tech] = value
    return state


def _report() -> str:
    web_lines = "\n".join(f"- 웹 근거 [웹 {url}]." for url in WEB_REFS)
    references = "\n".join(f"- Example. *자료*. {url}" for url in WEB_REFS)
    return f"""# KV cache 보고서

## SUMMARY

KIVI와 InfiniGen을 네 관점에서 비교했다 [논문 p.1].

## 1. 배경

고정 배경 설명이다.

## 2. 기술 선정

선정 과정이다.

## 3. 기술 개요

### KIVI

KIVI는 KV cache를 양자화한다 [논문 p.1].

### InfiniGen

InfiniGen은 KV cache를 오프로딩한다 [논문 p.1].

## 4. 관점별 평가

### 4.1 기술 성숙도

- **KIVI** — TRL 4
    - 공개 구현이 있다 [논문 p.2].
- **InfiniGen** — TRL 3
    - 공개 논문이 있다 [논문 p.2].

### 4.2 시장

- **KIVI** — 중
    - 시장 근거가 있다 [웹 {WEB_REFS[0]}].
- **InfiniGen** — 중
    - 시장 근거가 있다 [웹 {WEB_REFS[1]}].

### 4.3 이해관계자

- **KIVI** — 중립
    - 찬반 근거가 있다 [웹 {WEB_REFS[2]}].
- **InfiniGen** — 중립
    - 찬반 근거가 있다 [웹 {WEB_REFS[3]}].

### 4.4 도메인

- **KIVI** — 조건부
    - 온디바이스 조건이 있다 [웹 {WEB_REFS[4]}].
- **InfiniGen** — 조건부
    - 온디바이스 조건이 있다 [웹 {WEB_REFS[5]}].

{web_lines}

## 5. 시사점

두 기술은 서로 다른 자원 축을 포기한다 [논문 p.3].

## 6. 한계점

공개 정보 기반 분석이다.

## REFERENCE

- A(2024). *KIVI*. arXiv:2402.02750.
- B(2024). *InfiniGen*. arXiv:2406.19707.
{references}
"""


def test_run_passes_complete_grounded_report(tmp_path):
    path = tmp_path / "report.md"
    path.write_text(_report(), encoding="utf-8")
    state = _state()
    state["report_uri"] = path.as_posix()

    result = evaluator.run(state)

    assert result["llm_calls"] == 0
    assert result["eval_result"]["passed"]
    assert set(result["eval_result"]["items"]) == {"groundedness", "neutrality", "bias", "coverage"}
    assert result["eval_result"]["targets"] == []
    assert not ({"next", "retry", "rework_request"} & result.keys())


def test_missing_report_returns_structured_failure():
    result = evaluator.run(_state())

    assert result["llm_calls"] == 0 and not result["eval_result"]["passed"]
    assert result["eval_result"]["targets"] == ["report"]
    assert all(not item["passed"] for item in result["eval_result"]["items"].values())
    assert "report_uri 없음" in result["eval_result"]["feedback"]


def test_groundedness_fails_low_tag_coverage_and_high_inference():
    markdown = _report().replace(
        "KIVI와 InfiniGen을 네 관점에서 비교했다 [논문 p.1].",
        "KIVI와 InfiniGen을 네 관점에서 비교했다. 추가 판단에도 태그가 없다. "
        "시장 판단에도 태그가 없다. 제한된 자료에 따른 해석이다 [추론]. "
        "추가 검증이 필요한 해석이다 [추론]. 공개 정보만 사용한 해석이다 [추론].",
    )

    item = evaluator.groundedness(markdown)

    assert not item["passed"]
    assert "판단 문장 태그" in item["reason"] and "추론 단독" in item["reason"]


def test_groundedness_fails_when_body_citation_is_missing_from_reference():
    markdown = _report().replace(f"- Example. *자료*. {WEB_REFS[0]}\n", "")

    item = evaluator.groundedness(markdown)

    assert not item["passed"]
    assert WEB_REFS[0] in item["reason"] and "REFERENCE 누락" in item["reason"]


def test_bias_fails_when_one_web_source_dominates():
    state = deepcopy(_state())
    for key in ("market_eval", "stakeholder_eval", "domain_eval"):
        for value in state[key].values():
            value["evidence"][0]["ref"] = "https://same.example/article"

    item = evaluator.bias(state)

    assert not item["passed"]
    assert "same.example" in item["reason"] and "100%" in item["reason"]


def test_coverage_requires_four_perspectives_and_both_technologies():
    markdown = _report().replace("### 4.4 도메인", "### 도메인")

    item = evaluator.coverage(markdown)

    assert not item["passed"]
    assert "domain" in item["reason"] and "4.4:KIVI" in item["reason"] and "4.4:InfiniGen" in item["reason"]


def test_neutrality_rule_catches_direct_technology_recommendation():
    markdown = _report().replace(
        "두 기술은 서로 다른 자원 축을 포기한다 [논문 p.3].",
        "따라서 KIVI를 선택해야 한다 [추론].",
    )

    item = evaluator.neutrality(markdown)

    assert not item["passed"] and "추천" in item["reason"]
