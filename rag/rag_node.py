"""RAG 노드 공용 루틴 — 설계서 5.4.  [소유: A 박유진]

기술 조사(A)·도메인 평가(C) 워커가 같은 함수를 쓴다.

    ask(tech, question, node) -> (answer, evidence[], retrieval_entry, llm_calls)

고정 질문 → retriever(k=4, tech 필터) → 관련성 체크(재작성 ≤1, 전/후 둘 다 retrieval_log)
→ 답변 생성: 컨텍스트 [p.N] 태그, "문서에 없으면 없다고 답하라" → Faithfulness → citations
"""
from __future__ import annotations

GENERATOR_MODEL = "gpt-4.1-mini"


def ask(tech: str, question: str, node: str) -> dict:
    raise NotImplementedError
