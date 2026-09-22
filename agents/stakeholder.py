"""이해관계자 평가 에이전트 (웹) — 설계서 2장, 4.3 Rubric (찬·반 각 ≥2 — retry 시 '반대 근거 검색' 지시, 실패도 기록).  [소유: B 심준용]

출력 키: stakeholder_eval · citations · llm_calls
"""
from __future__ import annotations

from graph.state import GraphState


def run(state: GraphState) -> dict:
    raise NotImplementedError("stakeholder: 구현 예정 — docs/ROLES.md 참고")
