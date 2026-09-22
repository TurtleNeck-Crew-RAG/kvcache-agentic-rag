"""전처리 — 그림 1(a).  [소유: A 박유진]

PyMuPDFLoader (page 메타데이터 → [p.N] 인용)
→ RecursiveCharacterTextSplitter chunk 600 / overlap 100, metadata: chunk_id · tech · page
→ BGE-M3 dense + CacheBackedEmbeddings (키 = 청크 텍스트 해시, namespace = 모델명, data/cache/)
→ Chroma (data/index/chroma) + sparse 인덱스 (data/index/sparse_m3.json, BM25 는 chunks.jsonl 로 즉석 생성)

실행: uv run python -m rag.indexing   (data/papers/*.pdf 필요 → scripts/fetch_papers.sh)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.embeddings import CacheBackedEmbeddings  # langchain 1.x 는 classic 으로 이동
from langchain_classic.storage import LocalFileStore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.embeddings import MODEL_NAME, BGEM3Embeddings

CHUNK_SIZE, CHUNK_OVERLAP = 600, 100
PAPERS = {"KIVI": "data/papers/kivi.pdf", "InfiniGen": "data/papers/infinigen.pdf"}
ARXIV = {"KIVI": "2402.02750", "InfiniGen": "2406.19707"}

INDEX_DIR = Path("data/index")
CHROMA_DIR = INDEX_DIR / "chroma"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"       # sparse(BM25 · M3) 와 평가가 같이 씀
SPARSE_PATH = INDEX_DIR / "sparse_m3.json"
CACHE_DIR = Path("data/cache")
COLLECTION = "papers"


def load_chunks() -> list[Document]:
    """PDF 2편 → 페이지 단위 로드 → 600/100 분할. page 는 1부터 (인용 [p.N] 과 맞춤)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks: list[Document] = []
    for tech, path in PAPERS.items():
        pages = PyMuPDFLoader(path).load()
        for page in pages:
            page.metadata = {"tech": tech, "page": int(page.metadata["page"]) + 1, "arxiv": ARXIV[tech]}
        for i, doc in enumerate(splitter.split_documents(pages)):
            doc.metadata["chunk_id"] = f"{tech}-p{doc.metadata['page']}-{i:03d}"
            chunks.append(doc)
        print(f"{tech}: {len(pages)}p → 누적 {len(chunks)} chunks")
    return chunks


def get_embeddings() -> tuple[BGEM3Embeddings, CacheBackedEmbeddings]:
    base = BGEM3Embeddings()
    cached = CacheBackedEmbeddings.from_bytes_store(
        base, LocalFileStore(str(CACHE_DIR)), namespace=MODEL_NAME.replace("/", "_")
    )
    return base, cached


def build_index() -> None:
    t0 = time.time()
    chunks = load_chunks()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for d in chunks:
            f.write(json.dumps({"text": d.page_content, **d.metadata}, ensure_ascii=False) + "\n")

    base, cached = get_embeddings()
    n_cached = len(list(CACHE_DIR.glob("*"))) if CACHE_DIR.exists() else 0
    print(f"임베딩 캐시 히트 예상: {n_cached}/{len(chunks)}  (namespace={MODEL_NAME})")

    t1 = time.time()
    store = Chroma(
        collection_name=COLLECTION, embedding_function=cached, persist_directory=str(CHROMA_DIR)
    )
    if store._collection.count():
        store.delete_collection()
        store = Chroma(
            collection_name=COLLECTION, embedding_function=cached, persist_directory=str(CHROMA_DIR)
        )
    store.add_documents(chunks, ids=[d.metadata["chunk_id"] for d in chunks])
    t2 = time.time()

    sparse = base.encode_sparse([d.page_content for d in chunks])
    SPARSE_PATH.write_text(json.dumps(
        {d.metadata["chunk_id"]: w for d, w in zip(chunks, sparse, strict=True)}, ensure_ascii=False
    ))
    t3 = time.time()

    print(
        f"chunks={len(chunks)}  dense(Chroma)={t2 - t1:.1f}s  sparse(M3)={t3 - t2:.1f}s  "
        f"total={t3 - t0:.1f}s  → {CHROMA_DIR}, {SPARSE_PATH}"
    )


if __name__ == "__main__":
    build_index()
