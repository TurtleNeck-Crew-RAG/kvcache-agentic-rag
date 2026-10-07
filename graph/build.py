"""Graph 조립 — Supervisor (Hybrid 판정 + 결정론 게이트) · docs/ROLES.md 0절.  [소유: D 황재원]

START → supervisor ─route()─┬─ tech_research · market · stakeholder · domain ─→ assess ─→ supervisor
                            ├─ synthesis ─────────────────────────────────────────────→ supervisor
                            ├─ report ─→ evaluator ───────────────────────────────────→ supervisor
                            ├─ end_with_warning → END
                            └─ END

- 워커끼리 직접 잇지 않는다. 전부 판정 노드 또는 supervisor 로 복귀 — 다음 노드는 supervisor 만 정한다
- 매 턴 워커 1명 (fan-out 없음)
- 체크포인터를 넘기면 매 슈퍼스텝 State 가 저장된다. thread_id = trace_id (app.py)
"""
from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from graph.safe import safe
from graph.state import GraphState
from graph.supervisor import NEXT_NODES, end_with_warning, route, supervisor

Node = Callable[[dict], dict]
CELL_WORKERS = ("tech_research", "market", "stakeholder", "domain")   # → assess (충분성 판정)


def default_workers() -> dict[str, Node]:
    # 여기서 import — agents.* 는 rag.* (chromadb · FlagEmbedding) 를 끌어온다. 테스트는 workers 를 주입한다
    from agents import domain, market, report, stakeholder, synthesis, tech_research
    return {
        "tech_research": tech_research.run,
        "market": market.run,
        "stakeholder": stakeholder.run,
        "domain": domain.run,
        "synthesis": synthesis.run,
        "report": report.run,
    }


def default_judges() -> tuple[Node, Node]:
    # 판정 노드는 필수 — 없으면 import 에서 바로 실패한다. "판정 없이 통과"하는 자리 노드는 두지 않는다 (#120)
    from agents.evaluator import run as evaluator  # C
    from graph.sufficiency import assess  # A
    return assess, evaluator


def build_graph(workers: dict[str, Node] | None = None, assess: Node | None = None,
                evaluator: Node | None = None, checkpointer=None):
    workers = workers or default_workers()
    if assess is None or evaluator is None:
        a, e = default_judges()
        assess, evaluator = assess or a, evaluator or e

    g = StateGraph(GraphState)
    g.add_node("supervisor", supervisor)
    g.add_node("assess", safe("assess", assess))
    g.add_node("evaluator", safe("evaluator", evaluator))
    g.add_node("end_with_warning", end_with_warning)
    for name, fn in workers.items():
        g.add_node(name, safe(name, fn))         # 예외 → 실패 기록 + node_status failed (graph/safe.py)
        g.add_edge(name, "assess" if name in CELL_WORKERS else "evaluator" if name == "report" else "supervisor")

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route, list(NEXT_NODES))
    g.add_edge("assess", "supervisor")
    g.add_edge("evaluator", "supervisor")
    g.add_edge("end_with_warning", END)
    return g.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    # 그래프 그림 갱신: python -m graph.build → docs/images/graph_compiled.png
    app = build_graph()
    png = app.get_graph().draw_mermaid_png()
    with open("docs/images/graph_compiled.png", "wb") as f:
        f.write(png)
    print("docs/images/graph_compiled.png")
