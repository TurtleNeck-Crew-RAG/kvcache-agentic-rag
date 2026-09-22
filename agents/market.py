"""시장 평가 에이전트 (웹) — 설계서 2장, 4.2 Rubric (Tavily 만, RAG 없음, 기술별 독립 호출).  [소유: B 심준용]

출력 키: market_eval · citations · llm_calls
"""
from __future__ import annotations

from graph.state import GraphState


def run(state: GraphState) -> dict:
    raise NotImplementedError("market: 구현 예정 — docs/ROLES.md 참고")
