"""ask() 흐름 테스트 — 검색·LLM 을 전부 monkeypatch. 인덱스·API 키 불필요.
CI(langgraph+pytest 만 설치)에서는 의존성 없으면 skip."""
import pytest

pytest.importorskip("langchain_chroma")
pytest.importorskip("FlagEmbedding")

from langchain_core.documents import Document  # noqa: E402

import rag.rag_node as rn  # noqa: E402
from rag.rag_node import NO_EVIDENCE, RagAnswer, ask  # noqa: E402


def _docs(*ids):
    return [Document(page_content=f"text {i}", metadata={"chunk_id": i, "page": 9, "tech": "KIVI"}) for i in ids]


class _Retriever:
    def __init__(self, by_query):
        self.by_query = by_query

    def invoke(self, q):
        return self.by_query.get(q, [])


def _patch(monkeypatch, *, retriever, relevance, generate=None, faithful=True):
    monkeypatch.setattr(rn, "get_retriever", lambda *a, **k: retriever)
    monkeypatch.setattr(rn, "check_relevance", lambda q, docs: relevance(q))
    monkeypatch.setattr(rn, "rewrite_query", lambda q, *a: "REWRITTEN " + q)
    monkeypatch.setattr(rn, "_generate", lambda tech, q, docs: generate)
    monkeypatch.setattr(rn, "check_faithfulness", lambda *a: type("F", (), {"faithful": faithful, "unsupported": []})())


def test_happy_path_no_rewrite(monkeypatch):
    gen = RagAnswer(has_evidence=True, answer="2.6× 감소 [p.9]", evidence=[{"claim": "2.6× peak", "page": 9}])
    _patch(monkeypatch, retriever=_Retriever({"q": _docs("a", "b")}), relevance=lambda q: True, generate=gen)
    r = ask("KIVI", "q", node="t")
    assert r["answer"].endswith("[p.9]")
    assert r["evidence"][0] == {"claim": "2.6× peak", "tag": "논문", "ref": "2402.02750", "page": 9}
    assert r["citations"][0]["id_or_url"] == "arXiv:2402.02750"
    e = r["retrieval_entry"]
    assert e["relevance"] == "yes" and not e["rewritten"] and e["hits_before"] == ["a", "b"]
    assert r["llm_calls"] == 3          # judge1 + generator + judge2


def test_rewrite_then_success(monkeypatch):
    gen = RagAnswer(has_evidence=True, answer="A6000 [p.9]", evidence=[])
    ret = _Retriever({"q": _docs("x"), "REWRITTEN q": _docs("hit")})
    _patch(monkeypatch, retriever=ret, relevance=lambda q: q.startswith("REWRITTEN"), generate=gen)
    r = ask("InfiniGen", "q", node="t")
    e = r["retrieval_entry"]
    assert e["rewritten"] and e["query_after"] == "REWRITTEN q"
    assert e["hits_before"] == ["x"] and e["hits_after"] == ["hit"] and e["relevance"] == "yes"
    assert r["answer"] == "A6000 [p.9]" and r["llm_calls"] == 5


def test_no_evidence_after_rewrite(monkeypatch):
    _patch(monkeypatch, retriever=_Retriever({"q": _docs("x")}), relevance=lambda q: False)
    r = ask("KIVI", "q", node="t")
    assert r["answer"] == NO_EVIDENCE and r["evidence"] == [] and r["citations"] == []
    assert r["retrieval_entry"]["relevance"] == "no_evidence" and r["retrieval_entry"]["rewritten"]
    assert r["llm_calls"] == 3          # judge1 + rewrite + judge1


def test_generator_declines(monkeypatch):
    gen = RagAnswer(has_evidence=False, answer=NO_EVIDENCE)
    _patch(monkeypatch, retriever=_Retriever({"q": _docs("a")}), relevance=lambda q: True, generate=gen)
    r = ask("KIVI", "q", node="t")
    assert r["answer"] == NO_EVIDENCE and r["retrieval_entry"]["relevance"] == "no_evidence"
