"""도메인 평가 에이전트 (RAG + 웹) — 설계서 2장, 4.4 Rubric (1단계 사실 추출 RAG → 2단계 온디바이스 제약 판정 / HW 반례 웹검색 필수 / 다른 워커 결과 받지 않음).  [소유: C 민영은]

출력 키: domain_eval · citations · retrieval_log · llm_calls
"""
from __future__ import annotations

from graph.state import GraphState


def run(state: GraphState) -> dict:
    raise NotImplementedError("domain: 구현 예정 — docs/ROLES.md 참고")
