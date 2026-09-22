"""런타임 검색 — 그림 1(b).

hybrid_search(tech, dense_query, sparse_query) — dense + sparse RRF 0.5/0.5, k=4, `tech` 필터 (설계서 3.4).
**이중 질의**: dense 는 한국어 원 질의, sparse(BM25) 는 영어 번역 질의. 실측(#11, 20문항) 근거:
  dense-ko 0.55 / hybrid-m3-ko 0.45 / hybrid-bm25-ko 0.40  ← 한국어 질의에 sparse 를 섞으면 오히려 나빠짐
  dense-ko + bm25-en(mini 번역) 0.80                        ← 목표 Hit@4 ≥ 0.80 달성
설계서 3.4 초기값(BGE-M3 sparse)은 실측에서 뒤집힘 — 보고서 한계점 5 에 기록.

sparse 후보는 그대로 전환 가능 (평가 스크립트가 5 모드를 비교):
- "bm25" : rank_bm25 — data/index/chunks.jsonl 로 즉석 생성 (확정값)
- "m3"   : BGE-M3 learned sparse — data/index/sparse_m3.json (IDF 가 없어 빈출 토큰이 점수를 지배)
- None   : dense 단독

    from rag.retriever import hybrid_search, get_retriever
    docs = hybrid_search("KIVI", "피크 메모리는 얼마나 줄어드는가?", "KIVI peak memory reduction")
    docs = get_retriever("KIVI", sparse=None).invoke("...")     # 단일 질의 (평가용)
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
DEFAULT_SPARSE = "bm25"       # 실측(#11)으로 확정
RRF_C = 60
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


def rrf(ranked_lists: list[list[Document]], weights=WEIGHTS, k: int = TOP_K, c: int = RRF_C) -> list[Document]:
    """Reciprocal Rank Fusion — EnsembleRetriever 와 같은 식. 리스트마다 질의가 달라도 됨."""
    score: dict[str, float] = {}
    by_id: dict[str, Document] = {}
    for docs, w in zip(ranked_lists, weights, strict=True):
        for rank, d in enumerate(docs, 1):
            cid = d.metadata["chunk_id"]
            by_id[cid] = d
            score[cid] = score.get(cid, 0.0) + w / (c + rank)
    return [by_id[cid] for cid, _ in sorted(score.items(), key=lambda x: -x[1])[:k]]


def hybrid_search(
    tech: str, dense_query: str, sparse_query: str, *,
    sparse: SparseKind | None = DEFAULT_SPARSE, k: int = TOP_K, weights=WEIGHTS,
) -> list[Document]:
    """이중 질의 하이브리드 — dense(한국어 원 질의) + sparse(영어 질의) RRF. sparse=None 이면 dense 만."""
    dense_docs = DenseRetriever(tech=tech, k=2 * k).invoke(dense_query)
    if sparse is None:
        return dense_docs[:k]
    base, _ = get_embeddings()
    sp = SparseRetriever(tech=tech, kind=sparse, k=2 * k, query_encoder=base)
    return rrf([sp.invoke(sparse_query), dense_docs], weights, k)


def get_retriever(tech: str, sparse: SparseKind | None = DEFAULT_SPARSE, k: int = TOP_K) -> Runnable[str, list[Document]]:
    """단일 질의 Ensemble (평가·비교용). 런타임은 hybrid_search 를 쓴다."""
    dense = DenseRetriever(tech=tech, k=k)
    if sparse is None:
        return dense
    base, _ = get_embeddings()
    sp = SparseRetriever(tech=tech, kind=sparse, k=k, query_encoder=base)
    ens = EnsembleRetriever(retrievers=[sp, dense], weights=list(WEIGHTS))
    # EnsembleRetriever 는 두 리스트를 RRF 로 합친 전체(최대 2k)를 돌려주므로 k 로 자른다
    return ens | (lambda docs: docs[:k])
