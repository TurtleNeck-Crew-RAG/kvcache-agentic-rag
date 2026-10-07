"""[폐기 예정] RAG 과제 Dispatcher — Agent 과제는 graph/supervisor.py 로 대체됨 (#53).  [소유: D 황재원]

그래프에 연결돼 있지 않다. 남겨 둔 이유는 아직 이 모듈을 import 하는 다른 파트 코드뿐이다:
agents/stakeholder.py (LLM_BUDGET) · tests/test_stakeholder_worker.py · tests/test_synthesis.py (dispatcher()).
각 소유자가 rework_request 이행 PR 에서 graph.supervisor 로 옮기면 이 파일을 삭제한다.

--- 이하 RAG 과제 원문 ---
Dispatcher — 규칙표를 평가하는 유일한 곳 (설계서 5.3).

LLM 없음. State 검사만으로 next 와 retry 를 함께 기록한다.
LangGraph 는 노드 → 엣지 함수 순으로 실행하므로 판단과 카운터 증가를 같은 곳에 둔다.
"""
from __future__ import annotations

from langgraph.graph import END

from graph.state import GraphState
from graph.supervisor import (
    LLM_BUDGET,  # noqa: F401 — 상한은 supervisor 한 곳에서 (stakeholder 가 여기서 import)
)

MAX_RETRY = {"stake": 2, "synth": 2}
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
