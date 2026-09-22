"""RAG 노드 공용 루틴 — 설계서 5.4.  [소유: A 박유진]

기술 조사(A)·도메인 평가(C) 워커가 같은 함수를 쓴다.

    from rag.rag_node import ask
    r = ask("KIVI", "KIVI 적용 시 피크 메모리는 얼마나 줄어드는가?", node="tech_research")
    r["answer"]            # "... 2.6× 적은 피크 메모리 ... [p.1]"  또는 "논문에 근거 없음"
    r["evidence"]          # [Evidence{claim, tag:"논문", ref: arXiv id, page}]
    r["retrieval_entry"]   # RetrievalEntry — 재작성 전/후 둘 다
    r["citations"]         # [Ref] — 근거가 있을 때 논문 1건 (report 가 dedupe)
    r["llm_calls"]         # 이 호출에서 쓴 LLM 횟수
    r["faithful"]          # Judge 2 결과. False 면 unsupported 에 문장 목록

흐름: 검색(k=4, tech 필터) → Judge1 관련성 → (no) 재작성 1회 → 재검색 → Judge1
      → (no) "논문에 근거 없음" 기록 / (yes) 생성 [p.N] → Judge2 Faithfulness
"""
from __future__ import annotations

import re

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from agents._common import llm, render_prompt
from graph.state import Evidence, Ref, RetrievalEntry
from rag.judge import check_faithfulness, check_relevance, format_context, rewrite_query
from rag.retriever import SparseKind, get_retriever

NO_EVIDENCE = "논문에 근거 없음"

PAPERS: dict[str, Ref] = {   # 설계서 6장 REFERENCE 표기 예시 그대로
    "KIVI": {
        "type": "논문", "authors": "Liu, Z., Yuan, J., Jin, H. et al.", "year": "2024",
        "title": "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache",
        "venue": "Proceedings of the 41st International Conference on Machine Learning (ICML), PMLR 235",
        "id_or_url": "arXiv:2402.02750", "accessed": "2026-09-22",
    },
    "InfiniGen": {
        "type": "논문", "authors": "Lee, W., Lee, J., Seo, J., Sim, J.", "year": "2024",
        "title": "InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management",
        "venue": "18th USENIX Symposium on Operating Systems Design and Implementation (OSDI)",
        "id_or_url": "arXiv:2406.19707", "accessed": "2026-09-22",
    },
}


class EvidenceItem(BaseModel):
    claim: str = Field(description="컨텍스트에서 그대로 인용한 근거 문장")
    page: int


class RagAnswer(BaseModel):
    has_evidence: bool
    answer: str = Field(description="사실 문장마다 [p.N]. 근거 없으면 '논문에 근거 없음'")
    evidence: list[EvidenceItem] = Field(default_factory=list)


_BAD_TAG = re.compile(r"\[p\.(?!\d+\])[^\]]*\]")   # [p.4.1] 같은 절 번호 태그 — 페이지가 아니므로 제거


def _generate(tech: str, question: str, docs: list[Document]) -> RagAnswer:
    prompt = render_prompt("rag_generator", tech=tech, question=question, context=format_context(docs))
    gen = llm("generator").with_structured_output(RagAnswer).invoke(prompt)
    gen.answer = _BAD_TAG.sub("", gen.answer)
    pages = {d.metadata["page"] for d in docs}
    gen.evidence = [e for e in gen.evidence if e.page in pages]   # 컨텍스트에 없는 페이지 인용 제거
    return gen


def ask(tech: str, question: str, node: str, *, sparse: SparseKind | None = "m3", k: int = 4) -> dict:
    retriever = get_retriever(tech, sparse=sparse, k=k)
    calls = 0
    arxiv = PAPERS[tech]["id_or_url"].removeprefix("arXiv:")

    # ── 검색 + Judge 1 ──────────────────────────────────────────────
    docs = retriever.invoke(question)
    relevant = check_relevance(question, docs)
    calls += 1
    entry: RetrievalEntry = {
        "node": node, "tech": tech,
        "query_before": question, "query_after": None,
        "hits_before": [d.metadata["chunk_id"] for d in docs], "hits_after": None,
        "relevance": "yes" if relevant else "no", "rewritten": False,
    }

    # ── Loop 1: 재작성 1회 ──────────────────────────────────────────
    if not relevant:
        q2 = rewrite_query(question, tech, PAPERS[tech]["title"], docs)
        docs2 = retriever.invoke(q2)
        relevant = check_relevance(q2, docs2)
        calls += 2
        entry.update({
            "query_after": q2, "hits_after": [d.metadata["chunk_id"] for d in docs2],
            "rewritten": True, "relevance": "yes" if relevant else "no_evidence",
        })
        if relevant:
            docs = docs2

    if not relevant:   # 장치 5 — LLM 이 빈 검색을 메우지 못하게
        return {
            "answer": NO_EVIDENCE, "evidence": [], "retrieval_entry": entry,
            "citations": [], "llm_calls": calls, "faithful": True, "unsupported": [],
        }

    # ── 생성 + Judge 2 ──────────────────────────────────────────────
    gen = _generate(tech, question, docs)
    calls += 1
    if not gen.has_evidence:
        entry["relevance"] = "no_evidence"
        return {
            "answer": NO_EVIDENCE, "evidence": [], "retrieval_entry": entry,
            "citations": [], "llm_calls": calls, "faithful": True, "unsupported": [],
        }

    faith = check_faithfulness(question, gen.answer, docs)
    calls += 1
    evidence: list[Evidence] = [
        {"claim": e.claim, "tag": "논문", "ref": arxiv, "page": e.page} for e in gen.evidence
    ]
    return {
        "answer": gen.answer, "evidence": evidence, "retrieval_entry": entry,
        "citations": [PAPERS[tech]], "llm_calls": calls,
        "faithful": faith.faithful, "unsupported": faith.unsupported,
    }
