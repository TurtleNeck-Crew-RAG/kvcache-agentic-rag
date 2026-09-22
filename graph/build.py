"""Graph 조립 — Branching(fan-out/fan-in) + Loop (설계서 5.3).  [소유: D 황재원]

인덱싱 → 1 기술 조사 → 2 평가 3개 병렬 → 3 종합(+중립성 Judge) → 4 보고서
모든 워커는 END 로 직행하지 않고 dispatcher 로 되돌아온다.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents import domain, market, report, stakeholder, synthesis, tech_research
from graph.dispatcher import dispatcher, route
from graph.safe import safe
from graph.state import GraphState

WORKERS = {
    "tech_research": tech_research.run,
    "market": market.run,
    "stakeholder": stakeholder.run,
    "domain": domain.run,
    "synthesis": synthesis.run,     # 내부에서 중립성 Judge 호출 → neutrality 키 기록
    "report": report.run,
}


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("dispatcher", dispatcher)
    for name, fn in WORKERS.items():
        g.add_node(name, safe(name, fn))        # 예외 → 자기 키의 실패 기록 (graph/safe.py)
        g.add_edge(name, "dispatcher")          # 전부 dispatcher 로 수렴

    g.set_entry_point("dispatcher")
    g.add_conditional_edges("dispatcher", route, [*WORKERS.keys(), END])
    return g.compile()


if __name__ == "__main__":
    # 그래프 그림 갱신: python -m graph.build → docs/images/graph_compiled.png
    app = build_graph()
    png = app.get_graph().draw_mermaid_png()
    with open("docs/images/graph_compiled.png", "wb") as f:
        f.write(png)
    print("docs/images/graph_compiled.png")
