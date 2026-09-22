"""BGE-M3 임베딩 — dense + sparse 를 한 모델에서 (설계서 3.4 · 3.5).  [소유: A 박유진]

FlagEmbedding 의 BGEM3FlagModel 을 LangChain Embeddings 인터페이스로 감싼다.
- dense  : embed_documents / embed_query  → Chroma (CacheBackedEmbeddings 로 캐시)
- sparse : encode_sparse                  → learned sparse 인덱스 (BM25 대안, 실측 후 확정)

한 프로세스에 모델을 두 번 올리지 않도록 get_model() 은 싱글턴.
"""
from __future__ import annotations

from functools import lru_cache

import torch
from langchain_core.embeddings import Embeddings

MODEL_NAME = "BAAI/bge-m3"


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache
def get_model():
    from FlagEmbedding import BGEM3FlagModel  # import 가 느려서 지연 로딩

    return BGEM3FlagModel(MODEL_NAME, use_fp16=False, devices=_device())


class BGEM3Embeddings(Embeddings):
    """LangChain Embeddings 구현. 정규화된 1024차원 dense 벡터."""

    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = get_model().encode(
            texts, batch_size=self.batch_size, return_dense=True, return_sparse=False
        )
        return out["dense_vecs"].tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def encode_sparse(self, texts: list[str]) -> list[dict[str, float]]:
        """토큰 id → 가중치 dict. 질의·문서 양쪽에 같은 함수를 쓴다."""
        out = get_model().encode(
            texts, batch_size=self.batch_size, return_dense=False, return_sparse=True
        )
        return [{k: float(v) for k, v in w.items()} for w in out["lexical_weights"]]
