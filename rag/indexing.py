"""전처리 — 그림 1(a).  [소유: A 박유진]

PyMuPDFLoader (page 메타데이터 필수 → [p.N] 인용)
→ RecursiveCharacterTextSplitter chunk 600 / overlap 100, metadata: chunk_id · tech · page
→ BGE-M3 + CacheBackedEmbeddings (키 = 청크 텍스트 해시, namespace = 모델명, data/cache/)
→ Chroma (data/index/) + sparse 인덱스 (BGE-M3 sparse 초기값, BM25 와 비교 후 확정)

실행: python -m rag.indexing  (data/papers/*.pdf 가 있어야 함 → scripts/fetch_papers.sh)
"""
from __future__ import annotations

EMBED_MODEL = "BAAI/bge-m3"
CHUNK_SIZE, CHUNK_OVERLAP = 600, 100
PAPERS = {"KIVI": "data/papers/kivi.pdf", "InfiniGen": "data/papers/infinigen.pdf"}


def build_index() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    build_index()
