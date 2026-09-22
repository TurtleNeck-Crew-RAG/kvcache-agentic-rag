"""Dispatcher — 규칙표를 평가하는 유일한 곳 (설계서 5.3).  [소유: D 황재원]

LLM 없음. State 검사만으로 next 와 retry 를 함께 기록한다.
LangGraph 는 노드 → 엣지 함수 순으로 실행하므로 판단과 카운터 증가를 같은 곳에 둔다.
"""
from __future__ import annotations

from langgraph.graph import END

from graph.state import GraphState

MAX_RETRY = {"stake": 2, "synth": 2}
LLM_BUDGET = 150          # 안전망 — 초과 시 2'·3' 재호출만 중단, 3·4 는 진행. 설계서 5.0 은 100 → #29 로 150 (ask() 4회·기술 조사 46·도메인 46 실측)
MIN_NEGATIVES = 2         # 반대 근거 최소 건수 (설계서 5.5 장치 7)


def dispatcher(state: GraphState) -> dict:
    retry = dict(state.get("retry", {}))
    budget_ok = state.get("llm_calls", 0) <= LLM_BUDGET

    def go(nodes: list[str], bump: str | None = None) -> dict:
        if bump:
            retry[bump] = retry.get(bump, 0) + 1
        return {"next": nodes, "retry": retry}

    if not state.get("tech_summary"):                                   # 1
        return go(["tech_research"])

    pending = [w for w in ("market", "stakeholder", "domain") if not state.get(f"{w}_eval")]
    if pending:                                                         # 2 — 병렬
        return go(pending)

    neg = min(len(e["negatives"]) for e in state["stakeholder_eval"].values())
    if neg < MIN_NEGATIVES and retry.get("stake", 0) < MAX_RETRY["stake"] and budget_ok:
        return go(["stakeholder"], bump="stake")                        # 2'
    # 2'' — retry 소진: "반대 근거 확보 실패" 는 synthesis 워커가 한계점에 기록

    if not state.get("synthesis"):                                      # 3
        return go(["synthesis"])

    if (state.get("neutrality", {}).get("result") == "fail"
            and retry.get("synth", 0) < MAX_RETRY["synth"] and budget_ok):
        return go(["synthesis"], bump="synth")                          # 3'
    # 3'' — retry 소진: 위반 문장을 표시한 채 보고서로

    if not state.get("report_md"):                                      # 4
        return go(["report"])
    return go([END])


def route(state: GraphState) -> list[str]:
    """조건부 엣지 — state['next'] 만 반환."""
    return state["next"]
