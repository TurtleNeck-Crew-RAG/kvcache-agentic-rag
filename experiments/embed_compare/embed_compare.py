"""임베딩 후보 비교 — Hit Rate@K / MRR@K / 인덱싱 시간

사용법:
    uv add sentence-transformers pymupdf numpy
    python embed_compare.py --pdf data/kivi.pdf data/infinigen.pdf --eval eval_set.json

eval_set.json 항목: {"question_ko": ..., "question_en": ..., "gold": "정답 청크에 반드시 들어있는 문구"}
  - gold는 논문 원문에서 그대로 복사한 짧은 문구(숫자·고유명사 포함)로. 청크 어디에든 포함되면 정답.
결과: 모델 × 질의언어 별 표를 stdout + results.md 로 출력
"""
import argparse, json, time, re
from pathlib import Path

import numpy as np
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer

MODELS = [
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large",
    "BAAI/bge-base-en-v1.5",
    "nomic-ai/nomic-embed-text-v1.5",
]
# e5 / nomic 은 접두어를 요구함
PREFIX = {
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "nomic-ai/nomic-embed-text-v1.5": ("search_query: ", "search_document: "),
}


def load_chunks(pdfs, size=600, overlap=100):
    """페이지 텍스트 → 문자 기준 슬라이딩 청크. 파이프라인과 같은 크기."""
    chunks = []
    for pdf in pdfs:
        doc = fitz.open(pdf)
        for pno, page in enumerate(doc, 1):
            text = re.sub(r"\s+", " ", page.get_text()).strip()
            step = size - overlap
            for i in range(0, max(len(text) - overlap, 1), step):
                piece = text[i : i + size]
                if len(piece) > 50:
                    chunks.append({"text": piece, "src": Path(pdf).name, "page": pno})
    return chunks


def norm(s):
    return re.sub(r"\s+", " ", s).lower()


def evaluate(model_name, chunks, eval_set, k=4):
    q_pre, d_pre = PREFIX.get(model_name, ("", ""))
    model = SentenceTransformer(model_name, trust_remote_code=True)
    t0 = time.time()
    doc_emb = model.encode([d_pre + c["text"] for c in chunks], batch_size=16,
                           normalize_embeddings=True, show_progress_bar=False)
    index_sec = time.time() - t0
    docs_norm = [norm(c["text"]) for c in chunks]

    out = {"index_sec": index_sec}
    for lang in ("ko", "en"):
        hits, rr = 0, 0.0
        for item in eval_set:
            q = item[f"question_{lang}"]
            gold = norm(item["gold"])
            q_emb = model.encode([q_pre + q], normalize_embeddings=True)[0]
            top = np.argsort(-(doc_emb @ q_emb))[:k]
            for rank, idx in enumerate(top, 1):
                if gold in docs_norm[idx]:
                    hits += 1
                    rr += 1 / rank
                    break
        n = len(eval_set)
        out[lang] = {"hit": hits / n, "mrr": rr / n}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", nargs="+", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--models", nargs="*", default=MODELS)
    a = ap.parse_args()

    chunks = load_chunks(a.pdf)
    eval_set = json.loads(Path(a.eval).read_text())
    print(f"chunks={len(chunks)}  eval={len(eval_set)}  k={a.k}\n")

    # gold 문구가 실제로 어떤 청크에든 존재하는지 먼저 검증 (없으면 그 항목은 절대 못 맞춤)
    docs_norm = [norm(c["text"]) for c in chunks]
    for item in eval_set:
        if not any(norm(item["gold"]) in d for d in docs_norm):
            print(f"경고: gold 미존재: {item['gold'][:60]}")

    rows = []
    for m in a.models:
        r = evaluate(m, chunks, eval_set, a.k)
        rows.append((m, r))
        print(f"{m:40s} ko Hit@{a.k}={r['ko']['hit']:.2f} MRR={r['ko']['mrr']:.2f} | "
              f"en Hit@{a.k}={r['en']['hit']:.2f} MRR={r['en']['mrr']:.2f} | index {r['index_sec']:.0f}s")

    md = [f"| 모델 | ko Hit@{a.k} | ko MRR@{a.k} | en Hit@{a.k} | en MRR@{a.k} | 인덱싱(s) |", "|---|---|---|---|---|---|"]
    for m, r in rows:
        md.append(f"| `{m}` | {r['ko']['hit']:.2f} | {r['ko']['mrr']:.2f} | {r['en']['hit']:.2f} | {r['en']['mrr']:.2f} | {r['index_sec']:.0f} |")
    Path("results.md").write_text("\n".join(md))
    print("\n→ results.md 저장")


if __name__ == "__main__":
    main()
