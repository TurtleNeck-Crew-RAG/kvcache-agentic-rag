"""Dispatcher 규칙표 단위 테스트 — LLM·네트워크 불필요. (설계서 5.3)"""
import json
from pathlib import Path

from langgraph.graph import END

from graph.dispatcher import LLM_BUDGET, MAX_RETRY, dispatcher
from graph.state import init_state


def _eval(neg: int = 2) -> dict:
    return {"grade": "", "rationale": "", "positives": [], "negatives": ["x"] * neg,
            "evidence": [], "confidence": 1.0}


def test_rule1_tech_research_first():
    assert dispatcher(init_state({}, {}))["next"] == ["tech_research"]


def test_rule2_fanout_only_pending():
    s = init_state({}, {})
    s["tech_summary"] = {"KIVI": {}}
    s["market_eval"] = {"KIVI": _eval()}
    assert dispatcher(s)["next"] == ["stakeholder", "domain"]


def test_rule2p_retry_stakeholder_until_max():
    s = init_state({}, {})
    s["tech_summary"] = {"KIVI": {}}
    for k in ("market_eval", "stakeholder_eval", "domain_eval"):
        s[k] = {"KIVI": _eval(neg=1)}
    out = dispatcher(s)
    assert out["next"] == ["stakeholder"] and out["retry"] == {"stake": 1}
    s["retry"] = {"stake": MAX_RETRY["stake"]}
    assert dispatcher(s)["next"] == ["synthesis"]          # 2'' → 3


def test_rule3p_neutrality_loop_then_report():
    s = init_state({}, {})
    s["tech_summary"] = {"KIVI": {}}
    for k in ("market_eval", "stakeholder_eval", "domain_eval"):
        s[k] = {"KIVI": _eval()}
    s["synthesis"] = {"matrix": {}, "agreements": [], "conflicts": []}
    s["neutrality"] = {"result": "fail", "violations": ["추천"]}
    assert dispatcher(s)["next"] == ["synthesis"]
    s["retry"] = {"synth": MAX_RETRY["synth"]}
    assert dispatcher(s)["next"] == ["report"]             # 3'' → 4
    s["report_md"] = "# report"
    assert dispatcher(s)["next"] == [END]


def test_budget_stops_retries_only():
    s = init_state({}, {})
    s["tech_summary"] = {"KIVI": {}}
    s["llm_calls"] = LLM_BUDGET + 1
    for k in ("market_eval", "stakeholder_eval", "domain_eval"):
        s[k] = {"KIVI": _eval(neg=0)}
    assert dispatcher(s)["next"] == ["synthesis"]


# ---- fixtures 기반 — 실제 워커 출력 형식(tests/fixtures/*.json)으로 규칙표를 검증 ----
FIX = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    d = json.loads((FIX / name).read_text(encoding="utf-8"))
    d.pop("_note", None)
    return d


def _state_after_evals():
    s = init_state({}, {})
    s["tech_summary"] = _load("tech_summary.json")
    s.update(_load("evals.json"))
    return s


def test_fixtures_match_state_keys():
    s = _state_after_evals()
    for k in ("market_eval", "stakeholder_eval", "domain_eval"):
        assert set(s[k]) == {"KIVI", "InfiniGen"}, k
    for e in s["stakeholder_eval"].values():
        assert len(e["negatives"]) >= 2          # 규칙 2' 통과 조건


def test_fixtures_flow_to_synthesis_then_report():
    s = _state_after_evals()
    assert dispatcher(s)["next"] == ["synthesis"]           # 3
    s.update(_load("synthesis.json"))                       # C 출력 형식
    assert s["neutrality"]["result"] == "pass"
    assert dispatcher(s)["next"] == ["report"]              # 4
    s["report_md"] = "# report"
    assert dispatcher(s)["next"] == [END]


def test_synthesis_fixture_has_unused_citation_for_reference_filter():
    d = _load("synthesis.json")
    assert any("미인용" in c["authors"] for c in d["citations"])
