"""graph/safe.py — 워커 예외가 자기 키의 실패 기록으로 바뀌고, 형식이 State 와 맞는지. LLM 불필요."""
from langgraph.graph import END

from graph.dispatcher import MAX_RETRY, dispatcher
from graph.safe import fallback, safe
from graph.state import init_state


def _boom(state):
    raise NotImplementedError("stakeholder: 구현 예정")


def test_safe_returns_fallback_on_exception(capsys):
    out = safe("stakeholder", _boom)(init_state({}, {}))
    assert set(out) == {"stakeholder_eval"}
    for tech in ("KIVI", "InfiniGen"):
        e = out["stakeholder_eval"][tech]
        assert e["grade"] == "근거 없음" and "NotImplementedError" in e["rationale"] and e["negatives"] == []
    assert "[safe] stakeholder 실패" in capsys.readouterr().err


def test_safe_passes_through_normal_output():
    assert safe("market", lambda s: {"market_eval": {"KIVI": {}}})({}) == {"market_eval": {"KIVI": {}}}


def test_fallback_keys_match_state_for_every_worker():
    keys = {
        "tech_research": {"tech_summary"},
        "market": {"market_eval"},
        "stakeholder": {"stakeholder_eval"},
        "domain": {"domain_eval"},
        "synthesis": {"synthesis", "trl_estimate", "neutrality"},
        "report": {"report_md"},
    }
    for name, expect in keys.items():
        assert set(fallback(name, "x")) == expect, name
    assert fallback("domain", "x")["domain_eval"]["KIVI"]["verdict"] == "부적합"
    assert fallback("synthesis", "x")["neutrality"]["result"] == "pass"     # 3' 루프에 걸리지 않는다


def test_graph_reaches_end_when_all_workers_fail():
    """워커가 전부 예외를 내도 Dispatcher 규칙표대로 END 에 닿는다 — stakeholder 재호출은 상한(2)까지만."""
    s = init_state({}, {})
    visited = []
    for _ in range(20):
        nxt = dispatcher(s)
        s.update({"retry": nxt["retry"]})
        visited.append(nxt["next"])
        if nxt["next"] == [END]:
            break
        for name in nxt["next"]:
            s.update(fallback(name, "boom"))
    assert visited[-1] == [END]
    assert visited.count(["stakeholder"]) == MAX_RETRY["stake"]          # 2' 두 번 → 2'' 진행
    assert s["report_md"].startswith("# 보고서 생성 실패")
