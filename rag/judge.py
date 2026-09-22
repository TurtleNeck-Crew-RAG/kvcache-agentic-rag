"""Judge — 그림 1(b)·(c).  [소유: A 박유진]

Judge 1  관련성 structured yes/no → no 면 질문 재작성 1회 → 재실패 시 "논문에 근거 없음" 기록
Judge 2  Faithfulness — 생성이 컨텍스트에 근거하는지
모델: gpt-4.1-mini, temperature 0, Generator 와 별도 인스턴스·프롬프트 (설계서 3.6)
"""
from __future__ import annotations

JUDGE_MODEL = "gpt-4.1-mini"


def check_relevance(question: str, docs: list) -> bool:
    raise NotImplementedError


def rewrite_query(question: str) -> str:
    raise NotImplementedError


def check_faithfulness(answer: str, docs: list) -> bool:
    raise NotImplementedError
