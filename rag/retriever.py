"""런타임 검색 — 그림 1(b).  [소유: A 박유진]

EnsembleRetriever Sparse + Dense 0.5/0.5, k=4, `tech` 메타데이터 필터.
목표 Hit@4 ≥ 0.80 (dense 단독 0.65 → 하이브리드로 달성 여부 experiments/ 에서 확인)
"""
from __future__ import annotations

TOP_K = 4
WEIGHTS = (0.5, 0.5)   # (sparse, dense)


def get_retriever(tech: str):
    raise NotImplementedError
