"""RAG 파이프라인 — 설계서 3.4 · 3.7.  [소유: A 박유진]

전처리   loader → splitter → embedding(+cache) → store (Chroma + sparse)
런타임   retriever(Ensemble k=4, tech 필터) → judge1 관련성 → generator → judge2 faithfulness
"""
