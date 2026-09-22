# data/

| 경로 | 내용 | git |
|---|---|---|
| `papers/kivi.pdf`, `papers/infinigen.pdf` | RAG 코퍼스 — 풀 안 논문 2편 (33p). `bash scripts/fetch_papers.sh` 로 받음 | 제외 |
| `index/chroma/` | Chroma dense 인덱스 (BGE-M3, collection `papers`, metadata `tech`·`page`·`chunk_id`) | 제외 (재실행으로 재현) |
| `index/chunks.jsonl` · `index/sparse_m3.json` | 청크 원문+메타 (BM25 즉석 생성용) · BGE-M3 learned sparse 가중치 | 제외 |
| `cache/` | `CacheBackedEmbeddings` 로컬 파일 스토어 (키 = 청크 텍스트 해시, namespace = 모델명) | 제외 |

풀 밖 문서는 인덱싱하지 않는다 (과제 명세 "제시되는 풀에서 선택"). 반례·시장 자료는 Tavily 웹검색으로.
