"""최종 보고서 evaluator — 규칙층은 LLM 없이, Judge 층은 LLM 을 mock 해서 실행한다."""
from copy import deepcopy

import pytest

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


class FakeJudge:
    """with_structured_output(...).invoke(prompt) 를 흉내 낸다. 문제 id 만 지정하고 나머지는 통과로 채운다."""

    def __init__(self, claims=None, violations=None):
        self.claims = claims or {}          # id → ("overstated" | "unsupported", feedback)
        self.violations = violations or {}  # id → feedback
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        assert schema is evaluator.EvaluatorJudgeOutput
        return self

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        payload = evaluator.json.loads(prompt.split("## 검증 대상\n", 1)[1])
        return {
            "claims": [
                {"id": c["id"], "problem": self.claims.get(c["id"], ("none", ""))[0],
                 "feedback": self.claims.get(c["id"], ("none", ""))[1]}
                for c in payload["claims"]
            ],
            "neutrality": [
                {"id": n["id"], "violation": n["id"] in self.violations, "feedback": self.violations.get(n["id"], "")}
                for n in payload["neutrality"]
            ],
        }


@pytest.fixture
def fake_judge(monkeypatch):
    def install(**kwargs) -> FakeJudge:
        fake = FakeJudge(**kwargs)
        monkeypatch.setattr(evaluator, "llm", lambda role: fake)
        return fake
    return install


def _run(tmp_path, markdown: str, state=None) -> dict:
    path = tmp_path / "report.md"
    path.write_text(markdown, encoding="utf-8")
    state = state or _state()
    state["report_uri"] = path.as_posix()
    return evaluator.run(state)


def _claim_id(markdown: str, needle: str) -> str:
    claims = evaluator.judge_claims(markdown)
    return next(f"C{i}" for i, c in enumerate(claims, 1) if needle in c.text)


def test_run_passes_complete_grounded_report(tmp_path, fake_judge):
    judge = fake_judge()

    result = _run(tmp_path, _report())

    assert result["llm_calls"] == 1 and len(judge.prompts) == 1
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


def test_judge_claims_carry_section_worker_and_tech():
    claims = evaluator.judge_claims(_report())

    cells = {(c.worker, c.tech) for c in claims}
    assert ("market", "KIVI") in cells and ("domain", "InfiniGen") in cells
    assert ("tech_research", "KIVI") in cells
    assert all(evaluator.TAG_RE.sub("", c.text).strip(" .") for c in claims)
    assert len(claims) <= evaluator.MAX_JUDGE_CLAIMS


def test_judge_prompt_attaches_cell_evidence(tmp_path, fake_judge):
    judge = fake_judge()

    _run(tmp_path, _report())

    payload = evaluator.json.loads(judge.prompts[0].split("## 검증 대상\n", 1)[1])
    market_kivi = next(c for c in payload["claims"] if WEB_REFS[0] in c["text"])
    assert any(WEB_REFS[0] in ev for ev in market_kivi["evidence"])
    assert not any(WEB_REFS[1] in ev for ev in market_kivi["evidence"])


def test_overstated_claim_targets_report_with_feedback(tmp_path, fake_judge):
    markdown = _report()
    cid = _claim_id(markdown, "KIVI는 KV cache를 양자화한다")
    fake_judge(claims={cid: ("overstated", "양자화 비트 수와 실험 모델을 한정해 서술한다")})

    result = _run(tmp_path, markdown)["eval_result"]

    assert not result["passed"] and not result["items"]["groundedness"]["passed"]
    assert result["items"]["neutrality"]["passed"]
    assert result["targets"] == ["report"]
    assert "KIVI는 KV cache를 양자화한다" in result["feedback"]
    assert "근거보다 강한 주장" in result["feedback"] and "실험 모델을 한정" in result["feedback"]


def test_unsupported_claim_targets_source_worker_cell(tmp_path, fake_judge):
    markdown = _report()
    cid = _claim_id(markdown, f"시장 근거가 있다 [웹 {WEB_REFS[0]}]")
    fake_judge(claims={cid: ("unsupported", "시장 규모 수치의 출처를 다시 조사한다")})

    result = _run(tmp_path, markdown)["eval_result"]

    assert not result["items"]["groundedness"]["passed"]
    assert result["targets"] == ["market:KIVI"]
    assert "근거 없는 단정" in result["feedback"] and "(→ market:KIVI)" in result["feedback"]


def test_unsupported_claim_without_cell_falls_back_to_report(tmp_path, fake_judge):
    markdown = _report()
    cid = _claim_id(markdown, "네 관점에서 비교했다")   # SUMMARY · 두 기술 모두 언급 → 셀 없음
    fake_judge(claims={cid: ("unsupported", "근거를 붙이거나 문장을 삭제한다")})

    result = _run(tmp_path, markdown)["eval_result"]

    assert result["targets"] == ["report"]


def test_worker_target_precedes_report_target(tmp_path, fake_judge):
    markdown = _report().replace(
        "두 기술은 서로 다른 자원 축을 포기한다 [논문 p.3].",
        "온디바이스 환경에서는 InfiniGen이 더 적합한 기술이다 [논문 p.3].",
    )
    cid = _claim_id(markdown, f"온디바이스 조건이 있다 [웹 {WEB_REFS[5]}]")
    fake_judge(claims={cid: ("unsupported", "도메인 근거를 보강한다")}, violations={"N1": "조건별 병렬 서술로 바꾼다"})

    result = _run(tmp_path, markdown)["eval_result"]

    assert result["targets"] == ["domain:InfiniGen", "report"]


def test_contextual_neutrality_violation_missed_by_rule(tmp_path, fake_judge):
    sentence = "온디바이스 환경에서는 InfiniGen이 더 적합한 기술이다 [논문 p.3]."
    markdown = _report().replace("두 기술은 서로 다른 자원 축을 포기한다 [논문 p.3].", sentence)
    assert evaluator.neutrality(markdown)["passed"]          # 규칙 금지어에는 걸리지 않는다
    assert evaluator.neutrality_suspects(markdown) == [sentence]
    fake_judge(violations={"N1": "메모리·지연 조건별로 두 기술을 병렬 서술한다"})

    result = _run(tmp_path, markdown)["eval_result"]

    assert not result["passed"] and not result["items"]["neutrality"]["passed"]
    assert result["items"]["groundedness"]["passed"]
    assert result["targets"] == ["report"]
    assert "neutrality:" in result["feedback"] and "병렬 서술" in result["feedback"]


def test_neutral_comparison_judged_pass(tmp_path, fake_judge):
    markdown = _report().replace(
        "두 기술은 서로 다른 자원 축을 포기한다 [논문 p.3].",
        "저자들은 KIVI가 기준선 대비 처리량 면에서 우수하다고 보고한다 [논문 p.3].",
    )
    assert evaluator.neutrality(markdown)["passed"] and len(evaluator.neutrality_suspects(markdown)) == 1
    fake_judge()

    result = _run(tmp_path, markdown)["eval_result"]

    assert result["passed"]
    assert "중립 문맥" in result["items"]["neutrality"]["reason"]


def test_judge_skips_items_that_failed_rules(tmp_path, monkeypatch):
    markdown = _report().replace(
        "두 기술은 서로 다른 자원 축을 포기한다 [논문 p.3].",
        "따라서 KIVI를 선택해야 한다 [추론].",
    ).replace(f"- Example. *자료*. {WEB_REFS[0]}\n", "")

    def no_llm(role):
        raise AssertionError("규칙 fail 항목에는 LLM 을 호출하지 않는다")
    monkeypatch.setattr(evaluator, "llm", no_llm)

    result = _run(tmp_path, markdown)

    assert result["llm_calls"] == 0
    assert result["eval_result"]["targets"] == ["report"]
    assert not result["eval_result"]["items"]["groundedness"]["passed"]
    assert not result["eval_result"]["items"]["neutrality"]["passed"]


def test_judge_output_requires_feedback_for_problems():
    with pytest.raises(ValueError):
        evaluator.EvaluatorJudgeOutput.model_validate(
            {"claims": [{"id": "C1", "problem": "unsupported", "feedback": ""}], "neutrality": []}
        )
    with pytest.raises(ValueError):
        evaluator.EvaluatorJudgeOutput.model_validate(
            {"claims": [], "neutrality": [{"id": "N1", "violation": True, "feedback": " "}]}
        )


# ── #83: synthesis 중립성 Judge 에서 이전한 시나리오 ─────────────────────

def test_synthesis_recommendation_fails_then_neutral_rewrite_passes(tmp_path, fake_judge):
    """종합(5장)에 섞인 우열 표현은 보고서 뒤 evaluator 가 잡고, 중립적으로 다시 쓴 보고서는 통과한다."""
    original = "두 기술은 서로 다른 자원 축을 포기한다 [논문 p.3]."
    biased = _report().replace(original, "시장 관점에서 KIVI가 더 우수하다 [추론].")
    fake_judge()

    first = _run(tmp_path, biased)

    assert not first["eval_result"]["items"]["neutrality"]["passed"]
    assert first["eval_result"]["targets"] == ["report"]
    assert "neutrality:" in first["eval_result"]["feedback"]

    rewritten = _run(tmp_path, _report())

    assert rewritten["eval_result"]["passed"] and rewritten["llm_calls"] == 1


def test_trl_parallel_statement_is_not_a_neutrality_suspect():
    markdown = _report().replace(
        "두 기술은 서로 다른 자원 축을 포기한다 [논문 p.3].",
        "KIVI는 TRL 4, InfiniGen은 TRL 3으로 추정된다 [추론]. 두 기술의 실험 조건이 다르다 [논문 p.3].",
    )

    assert evaluator.neutrality(markdown)["passed"]
    assert evaluator.neutrality_suspects(markdown) == []


# ── #90: arXiv URL 정규화 · 태그 비율 분모 · 항목별 target ─────────────────

def test_arxiv_urls_in_body_match_normalized_reference_ids():
    markdown = _report().replace(
        "KIVI는 KV cache를 양자화한다 [논문 p.1].",
        "KIVI는 KV cache를 양자화한다 [웹 https://arxiv.org/html/2402.02750v2]. "
        "InfiniGen도 인용한다 [웹 https://arxiv.org/abs/2406.19707] [웹 https://arxiv.org/pdf/2406.19707v1].",
    )

    item = evaluator.groundedness(markdown)

    assert "REFERENCE 누락" not in item["reason"] and "본문 미인용" not in item["reason"]


def test_arxiv_url_missing_from_reference_is_reported_as_id():
    markdown = _report().replace(
        "KIVI는 KV cache를 양자화한다 [논문 p.1].",
        "KIVI는 KV cache를 양자화한다 [웹 https://arxiv.org/pdf/2604.05012].",
    )

    item = evaluator.groundedness(markdown)

    assert not item["passed"] and "REFERENCE 누락: arXiv:2604.05012" in item["reason"]


def test_trailing_tag_after_period_belongs_to_previous_sentence():
    markdown = _report().replace(
        "KIVI는 KV cache를 양자화한다 [논문 p.1].",
        f"KIVI는 KV cache를 양자화한다. [웹 {WEB_REFS[0]}]",
    )

    units = evaluator.claim_units(markdown)

    assert f"KIVI는 KV cache를 양자화한다. [웹 {WEB_REFS[0]}]" in units
    assert f"[웹 {WEB_REFS[0]}]" not in units
    assert evaluator.groundedness(markdown)["passed"]


def test_absence_records_and_follow_up_items_are_not_claims():
    markdown = _report().replace(
        "두 기술은 서로 다른 자원 축을 포기한다 [논문 p.3].",
        "두 기술은 서로 다른 자원 축을 포기한다 [논문 p.3].\n\n"
        "- 근거 없음\n- 판단에 필요한 추가 확인 항목은 InfiniGen의 실측 결과다.",
    )

    units = evaluator.claim_units(markdown)

    assert "근거 없음" not in units
    assert not any(u.startswith("판단에 필요한 추가 확인 항목") for u in units)


def test_bias_missing_negative_targets_that_cell():
    state = _state()
    state["stakeholder_eval"]["InfiniGen"]["negatives"] = []

    result = evaluator.evaluate(_report(), state)

    assert not result["items"]["bias"]["passed"]
    assert result["targets"] == ["stakeholder:InfiniGen"]


def test_bias_source_concentration_targets_heaviest_cells_first():
    state = deepcopy(_state())
    for key, tech in (("market_eval", "KIVI"), ("domain_eval", "InfiniGen")):
        state[key][tech]["evidence"] = [
            {"claim": "근거", "tag": "웹", "ref": "https://same.example/a", "page": None}
        ] * (3 if key == "market_eval" else 2)

    result = evaluator.evaluate(_report(), state)

    assert "same.example" in result["items"]["bias"]["reason"]
    assert result["targets"][:2] == ["market:KIVI", "domain:InfiniGen"]
    assert "report" not in result["targets"]


def test_bias_counts_arxiv_urls_as_one_paper_source():
    assert evaluator._source_key("https://arxiv.org/html/2402.02750v2") == "arXiv:2402.02750"
    assert evaluator._source_key("https://arxiv.org/abs/2402.02750") == "arXiv:2402.02750"
    assert evaluator._source_key("https://www.github.com/jy-yuan/KIVI") == "github.com"


def test_coverage_gap_targets_worker_only_when_cell_payload_missing():
    markdown = _report().replace("- **InfiniGen** — 조건부\n    - 온디바이스 조건이 있다", "- 온디바이스 조건이 있다")
    rendered_gap = evaluator.evaluate(markdown, _state())
    state = _state()
    state["domain_eval"].pop("InfiniGen")
    source_gap = evaluator.evaluate(markdown, state)

    assert "4.4:InfiniGen" in rendered_gap["items"]["coverage"]["reason"]
    assert rendered_gap["targets"] == ["report"]                  # State 에는 있음 → 렌더링 문제
    assert source_gap["targets"][0] == "domain:InfiniGen"         # State 에도 없음 → 워커 재조사


def test_rule_targets_put_cells_before_report():
    state = _state()
    state["market_eval"]["KIVI"]["negatives"] = []
    markdown = _report().replace("두 기술은 서로 다른 자원 축을 포기한다 [논문 p.3].", "따라서 KIVI를 선택해야 한다 [추론].")

    result = evaluator.evaluate(markdown, state)

    assert result["targets"] == ["market:KIVI", "report"]


# ── #117: groundedness 실패는 고칠 수 있는 곳으로 — 4장은 셀 워커, 나머지 장은 report ──────────

UNTAGGED = "근거 없이 단정한 문장이다. 또 다른 단정 문장이다. 세 번째 단정 문장이다."


def test_untagged_rendered_cell_sentences_target_that_worker():
    markdown = _report().replace(f"시장 근거가 있다 [웹 {WEB_REFS[0]}].", f"{UNTAGGED} 시장 근거가 있다 [웹 {WEB_REFS[0]}].")

    result = evaluator.evaluate(markdown, _state())

    assert not result["items"]["groundedness"]["passed"]
    assert result["targets"] == ["market:KIVI"]                  # 보고서를 다시 써도 4장은 안 바뀐다
    assert "market:KIVI 3건" in result["items"]["groundedness"]["reason"]


def test_report_chapter_problems_target_report_and_cells_come_first():
    markdown = (
        _report()
        .replace("KIVI와 InfiniGen을 네 관점에서 비교했다 [논문 p.1].", UNTAGGED)
        .replace(f"온디바이스 조건이 있다 [웹 {WEB_REFS[5]}].", f"{UNTAGGED} 조건 [웹 {WEB_REFS[5]}].")
    )

    result = evaluator.evaluate(markdown, _state())

    assert result["targets"] == ["domain:InfiniGen", "report"]


def test_chapter3_overview_is_rewritten_by_report_not_tech_research():
    markdown = _report().replace("KIVI는 KV cache를 양자화한다 [논문 p.1].", UNTAGGED + " 양자화한다 [추론].")

    result = evaluator.evaluate(markdown, _state())

    assert not result["items"]["groundedness"]["passed"]
    assert result["targets"] == ["report"]


def test_perspective_4_1_and_inference_only_map_to_cells():
    markdown = (
        _report()
        .replace("공개 구현이 있다 [논문 p.2].", "공개 구현이 있을 것이다 [추론]. 재현이 쉬울 것이다 [추론].")
        .replace(f"찬반 근거가 있다 [웹 {WEB_REFS[3]}].", "찬반이 갈릴 것이다 [추론]. 반대가 많을 것이다 [추론].")
    )

    targets = evaluator.evaluate(markdown, _state())["targets"]

    assert set(targets) == {"tech_research:KIVI", "stakeholder:InfiniGen"}


def test_labelled_absence_record_is_not_a_claim():
    markdown = _report().replace(
        f"- **KIVI** — 중\n    - 시장 근거가 있다 [웹 {WEB_REFS[0]}].",
        f"- **KIVI** — 중\n    - 채택: 근거 없음\n    - 시장 근거가 있다 [웹 {WEB_REFS[0]}].",
    )

    units = evaluator.claim_units(markdown)

    assert "채택: 근거 없음" not in units
    assert evaluator.groundedness(markdown)["passed"]


def test_judge_overstated_rendered_claim_targets_cell(tmp_path, fake_judge):
    markdown = _report()
    cid = _claim_id(markdown, f"시장 근거가 있다 [웹 {WEB_REFS[0]}]")
    fake_judge(claims={cid: ("overstated", "채택 범위를 근거의 실험 조건으로 한정한다")})

    result = _run(tmp_path, markdown)["eval_result"]

    assert result["targets"] == ["market:KIVI"]
    assert "근거보다 강한 주장" in result["feedback"] and "(→ market:KIVI)" in result["feedback"]
