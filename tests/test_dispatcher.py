"""Dispatcher 규칙표 단위 테스트 — LLM·네트워크 불필요. (설계서 5.3)"""
from langgraph.graph import END

from graph.dispatcher import MAX_RETRY, dispatcher
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
    s["llm_calls"] = 101
    for k in ("market_eval", "stakeholder_eval", "domain_eval"):
        s[k] = {"KIVI": _eval(neg=0)}
    assert dispatcher(s)["next"] == ["synthesis"]
