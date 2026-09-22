"""평가 종합 에이전트 + 중립성 Judge — 설계서 2장, 4.1 TRL · 4.5 종합 Rubric, 5.5 장치 8 (관점×기술 매트릭스, 일치/상충, 우열·추천 표현 탐지 → neutrality).  [소유: C 민영은]

출력 키: synthesis · trl_estimate · neutrality · llm_calls
"""
from __future__ import annotations

from graph.state import GraphState


def run(state: GraphState) -> dict:
    raise NotImplementedError("synthesis: 구현 예정 — docs/ROLES.md 참고")
