"""graph/state.py — ROLES.md 4절 계약 · 리듀서. LLM 불필요."""
from typing import get_type_hints

from langgraph.graph import END, START, StateGraph

from graph.state import GraphState, init_state, merge, merge_by_tech

CONTRACT = {
    # 페이로드
    "domain", "selected", "tech_summary", "market_eval", "stakeholder_eval", "domain_eval",
    "trl_estimate", "synthesis", "report_uri", "citations",
    # 판정
    "sufficiency", "eval_result",
    # 제어
    "trace_id", "next", "rework_request", "step_count", "max_steps", "retry", "eval_attempts",
    "llm_calls", "status", "node_status", "errors", "last_error",
}


def test_contract_keys_present():
    assert CONTRACT <= set(get_type_hints(GraphState, include_extras=True))


def test_init_state_control_defaults():
    s = init_state({}, {}, trace_id="t-1", max_steps=7)
    assert s["trace_id"] == "t-1" and s["max_steps"] == 7
    assert s["next"] == "" and s["step_count"] == 0 and s["status"] == "RUNNING"
    assert s["rework_request"] is None and s["eval_result"] is None and s["last_error"] is None
    assert s["retry"] == {} and s["node_status"] == {} and s["errors"] == []


def test_merge_by_tech_keeps_other_tech():
    old = {"KIVI": {"grade": "상"}, "InfiniGen": {"grade": "근거 없음"}}
    assert merge_by_tech(old, {"InfiniGen": {"grade": "중"}}) == {"KIVI": {"grade": "상"}, "InfiniGen": {"grade": "중"}}
    assert merge(None, {"a": 1}) == {"a": 1}


def _fanout_graph():
    """두 노드가 같은 슈퍼스텝에 node_status · errors · sufficiency · step_count 에 쓴다 — 리듀서가 없으면 InvalidUpdateError."""
    def a(_):
        return {"node_status": {"a": "ok"}, "errors": [], "sufficiency": {"market:KIVI": {"rule": "pass"}},
                "step_count": 1, "market_eval": {"KIVI": {"grade": "상"}}}

    def b(_):
        return {"node_status": {"b": "failed"}, "errors": [{"node": "b", "type": "E", "message": "", "ts": ""}],
                "sufficiency": {"market:InfiniGen": {"rule": "fail"}}, "step_count": 1,
                "market_eval": {"InfiniGen": {"grade": "근거 없음"}}}

    g = StateGraph(GraphState)
    g.add_node("a", a)
    g.add_node("b", b)
    g.add_edge(START, "a")
    g.add_edge(START, "b")
    g.add_edge("a", END)
    g.add_edge("b", END)
    return g.compile()


def test_reducers_allow_same_superstep_writes():
    out = _fanout_graph().invoke(init_state({}, {}))
    assert out["node_status"] == {"a": "ok", "b": "failed"}
    assert set(out["sufficiency"]) == {"market:KIVI", "market:InfiniGen"}
    assert set(out["market_eval"]) == {"KIVI", "InfiniGen"}
    assert out["step_count"] == 2 and len(out["errors"]) == 1
