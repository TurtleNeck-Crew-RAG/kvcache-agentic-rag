"""보고서 생성 에이전트 — 설계서 6장 목차 (SUMMARY ½p → 1~6장 → REFERENCE — citations 에서 인용된 항목만, 지정 표기 형식).  [소유: D 황재원]

출력 키: report_md (+ outputs/report/*.md → PDF)
"""
from __future__ import annotations

from graph.state import GraphState


def run(state: GraphState) -> dict:
    raise NotImplementedError("report: 구현 예정 — docs/ROLES.md 참고")
