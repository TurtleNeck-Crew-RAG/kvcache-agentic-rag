"""런타임 검색 — 그림 1(b).  [소유: A 박유진]

EnsembleRetriever(sparse + dense, 0.5/0.5, RRF), k=4, `tech` 메타데이터 필터 (설계서 3.4).

sparse 후보 2개를 같은 인터페이스로 두고 실측(#11)으로 확정:
- "m3"   : BGE-M3 learned sparse — data/index/sparse_m3.json (초기값, 다국어 표면형 차이를 용어 확장으로 흡수)
- "bm25" : rank_bm25 — data/index/chunks.jsonl 로 즉석 생성 (표면형 정확 일치)
- None   : dense 단독 (베이스라인)

    from rag.retriever import get_retriever
    docs = get_retriever("KIVI").invoke("피크 메모리는 얼마나 줄어드는가?")
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Literal

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from pydantic import Field

from rag.indexing import CHROMA_DIR, CHUNKS_PATH, COLLECTION, SPARSE_PATH, get_embeddings

TOP_K = 4
WEIGHTS = (0.5, 0.5)          # (sparse, dense)
SparseKind = Literal["m3", "bm25"]


# ── 인덱스 로딩 (프로세스당 1회) ────────────────────────────────────────────
@lru_cache
def _chunks() -> list[Document]:
    rows = [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
    return [Document(page_content=r.pop("text"), metadata=r) for r in rows]


@lru_cache
def _sparse_weights() -> dict[str, dict[str, float]]:
    return json.loads(SPARSE_PATH.read_text(encoding="utf-8"))


@lru_cache
def _chroma() -> Chroma:
    _, cached = get_embeddings()
    return Chroma(collection_name=COLLECTION, embedding_function=cached, persist_directory=str(CHROMA_DIR))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*|[가-힣]+", text.lower())


@lru_cache
def _bm25(tech: str):
    from rank_bm25 import BM25Okapi

    docs = [d for d in _chunks() if d.metadata["tech"] == tech]
    return docs, BM25Okapi([_tokenize(d.page_content) for d in docs])


# ── Retriever 구현 ──────────────────────────────────────────────────────────
class SparseRetriever(BaseRetriever):
    """BGE-M3 learned sparse 또는 BM25. tech 필터는 chunk 메타데이터로."""

    tech: str
    kind: SparseKind = "m3"
    k: int = TOP_K
    query_encoder: object | None = Field(default=None, exclude=True)   # m3 일 때 BGEM3Embeddings

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        if self.kind == "bm25":
            docs, bm25 = _bm25(self.tech)
            scores = bm25.get_scores(_tokenize(query))
        else:
            q = self.query_encoder.encode_sparse([query])[0]
            weights = _sparse_weights()
            docs = [d for d in _chunks() if d.metadata["tech"] == self.tech]
            scores = [
                sum(w * weights[d.metadata["chunk_id"]].get(tok, 0.0) for tok, w in q.items())
                for d in docs
            ]
        ranked = sorted(zip(scores, docs, strict=True), key=lambda x: -x[0])
        return [d for s, d in ranked[: self.k] if s > 0]


class DenseRetriever(BaseRetriever):
    tech: str
    k: int = TOP_K

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return _chroma().similarity_search(query, k=self.k, filter={"tech": self.tech})


def get_retriever(tech: str, sparse: SparseKind | None = "m3", k: int = TOP_K) -> Runnable[str, list[Document]]:
    """설계서 3.4 — Ensemble sparse+dense 0.5/0.5, k=4, tech 필터.

    sparse=None 이면 dense 단독 (실측 베이스라인).
    """
    dense = DenseRetriever(tech=tech, k=k)
    if sparse is None:
        return dense
    base, _ = get_embeddings()
    sp = SparseRetriever(tech=tech, kind=sparse, k=k, query_encoder=base)
    ens = EnsembleRetriever(retrievers=[sp, dense], weights=list(WEIGHTS))
    # EnsembleRetriever 는 두 리스트를 RRF 로 합친 전체(최대 2k)를 돌려주므로 k 로 자른다
    return ens | (lambda docs: docs[:k])
