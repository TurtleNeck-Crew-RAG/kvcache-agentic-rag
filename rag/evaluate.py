"""검색·생성 평가 — 설계서 3.4 평가 행 · 3.5 해석 4·5 · 6장 한계점 5·6.  [소유: A 박유진]

검색  Hit Rate@4 · MRR@4 — experiments/embed_compare/eval_set.json 20문항(논문당 10, ko/en 쌍)
      단일 질의 모드: dense / sparse-m3 / sparse-bm25 / hybrid-m3 / hybrid-bm25  (ko · en 각각)
      이중 질의 모드: dual-bm25 / dual-m3 — dense 는 ko 원 질의, sparse 는 LLM 번역 영어 질의 (런타임과 동일 경로)
생성  RAGAS Faithfulness · ResponseRelevancy · LLMContextPrecisionWithoutReference — ask() 20문항(ko)
재작성 효과  outputs/retrieval_log.json 이 있으면 rewritten 건의 전/후 relevance 집계 (한계점 6)

실행: uv run python -m rag.evaluate [--no-ragas]   → outputs/eval.json + stdout 표 (README 에 옮김)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from rag.indexing import CHUNKS_PATH
from rag.retriever import TOP_K, SparseRetriever, get_retriever, hybrid_search

EVAL_SET = Path("experiments/embed_compare/eval_set.json")
OUT = Path("outputs/eval.json")
MODES = {
    "dense": ("dense", None),
    "sparse-m3": ("sparse", "m3"),
    "sparse-bm25": ("sparse", "bm25"),
    "hybrid-m3": ("hybrid", "m3"),
    "hybrid-bm25": ("hybrid", "bm25"),
    "dual-bm25": ("dual", "bm25"),
    "dual-m3": ("dual", "m3"),
}
TRANSLATIONS = Path("outputs/eval_translations.json")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def load_eval_set() -> list[dict]:
    qs = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    for i, q in enumerate(qs):
        q["tech"] = "KIVI" if i < len(qs) // 2 else "InfiniGen"     # 앞 10 = KIVI, 뒤 10 = InfiniGen
    return qs


def _gold_chunks(qs: list[dict]) -> dict[int, set[str]]:
    """gold 문구를 포함한 chunk_id 집합 (embed_compare 와 같은 판정)."""
    rows = [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
    out = {}
    for i, q in enumerate(qs):
        g = _norm(q["gold"])
        out[i] = {r["chunk_id"] for r in rows if r["tech"] == q["tech"] and g in _norm(r["text"])}
        if not out[i]:
            print(f"  ⚠ gold 미발견 (문항 {i}): {q['gold']!r}", file=sys.stderr)
    return out


def _retriever(mode: str, tech: str):
    kind, sparse = MODES[mode]
    if kind == "sparse":
        from rag.indexing import get_embeddings
        base, _ = get_embeddings()
        return SparseRetriever(tech=tech, kind=sparse, k=TOP_K, query_encoder=base)
    return get_retriever(tech, sparse=sparse, k=TOP_K)


def _translations(qs: list[dict]) -> list[str]:
    """ko 질의 → 영어 (rag.judge.translate_query, mini). 한 번 만들고 캐시."""
    if TRANSLATIONS.exists():
        cached = json.loads(TRANSLATIONS.read_text(encoding="utf-8"))
        if len(cached) == len(qs):
            return cached
    from rag.judge import translate_query
    tr = [translate_query(q["question_ko"]) for q in qs]
    TRANSLATIONS.parent.mkdir(exist_ok=True)
    TRANSLATIONS.write_text(json.dumps(tr, ensure_ascii=False, indent=1), encoding="utf-8")
    return tr


def eval_retrieval(qs: list[dict], modes=tuple(MODES)) -> dict:
    gold = _gold_chunks(qs)
    result = {}
    for mode in modes:
        kind, sparse = MODES[mode]
        langs = ("ko",) if kind == "dual" else ("ko", "en")
        tr = _translations(qs) if kind == "dual" else None
        for lang in langs:
            hits, rr, misses = 0, 0.0, []
            for i, q in enumerate(qs):
                if kind == "dual":
                    docs = hybrid_search(q["tech"], q["question_ko"], tr[i], sparse=sparse, k=TOP_K)
                else:
                    docs = _retriever(mode, q["tech"]).invoke(q[f"question_{lang}"])
                ids = [d.metadata["chunk_id"] for d in docs]
                rank = next((r for r, cid in enumerate(ids, 1) if cid in gold[i]), None)
                if rank:
                    hits += 1
                    rr += 1 / rank
                else:
                    misses.append(i)
            n = len(qs)
            label = f"{mode}/{lang}" + ("(dense)+en(sparse)" if kind == "dual" else "")
            result[label] = {"hit@4": round(hits / n, 2), "mrr@4": round(rr / n, 2), "miss": misses}
            print(f"  {label:28s} Hit@4={hits / n:.2f}  MRR@4={rr / n:.2f}  miss={misses}")
    return result


def eval_ragas(qs: list[dict]) -> dict:
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy

    from rag.embeddings import BGEM3Embeddings
    from rag.rag_node import NO_EVIDENCE, ask

    samples, skipped, calls = [], 0, 0
    for q in qs:
        r = ask(q["tech"], q["question_ko"], node="eval")
        calls += r["llm_calls"]
        if r["answer"] == NO_EVIDENCE:
            skipped += 1                         # 근거 없음은 생성 품질 평가 대상이 아님 (검색 실패로 집계)
            continue
        samples.append(SingleTurnSample(user_input=q["question_ko"], response=r["answer"], retrieved_contexts=r["contexts"]))
    judge = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    emb = LangchainEmbeddingsWrapper(BGEM3Embeddings())
    res = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[Faithfulness(llm=judge), ResponseRelevancy(llm=judge, embeddings=emb), LLMContextPrecisionWithoutReference(llm=judge)],
    )
    df = res.to_pandas()
    out = {c: round(float(df[c].mean()), 3) for c in ("faithfulness", "answer_relevancy", "llm_context_precision_without_reference") if c in df}
    out |= {"n": len(samples), "no_evidence": skipped, "llm_calls_ask": calls}
    print(f"  RAGAS (n={len(samples)}, 근거없음 {skipped}): {out}")
    return out


def rewrite_effect(path: Path = Path("outputs/retrieval_log.json")) -> dict | None:
    if not path.exists():
        return None
    log = json.loads(path.read_text(encoding="utf-8"))
    rw = [e for e in log if e["rewritten"]]
    rescued = sum(e["relevance"] == "yes" for e in rw)
    out = {"total": len(log), "rewritten": len(rw), "rescued": rescued, "still_no_evidence": len(rw) - rescued}
    print(f"  재작성 효과: {len(log)}건 중 재작성 {len(rw)} → 구제 {rescued}, 실패 {len(rw) - rescued}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ragas", action="store_true")
    ap.add_argument("--modes", nargs="*", default=list(MODES))
    args = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(".env")

    qs = load_eval_set()
    print(f"[검색] {len(qs)}문항, k={TOP_K}")
    out = {"retrieval": eval_retrieval(qs, args.modes)}
    print("[재작성]")
    out["rewrite"] = rewrite_effect()
    if not args.no_ragas:
        print("[생성] RAGAS")
        out["ragas"] = eval_ragas(qs)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
