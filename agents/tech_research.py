"""기술 조사 에이전트 (RAG) — 설계서 2장, 5.4 (고정 질문 5: 개요·방법·수치·한계·적용조건 / 기술별 독립 호출 / retriever tech 필터).  [소유: A 박유진]

출력 키: tech_summary · citations · retrieval_log · llm_calls
"""
from __future__ import annotations

from graph.state import GraphState


def run(state: GraphState) -> dict:
    raise NotImplementedError("tech_research: 구현 예정 — docs/ROLES.md 참고")
