"""기술 조사 에이전트 (RAG) — 설계서 2장, 5.4.

고정 질문 5(개요·메커니즘·수치·한계·적용조건) × 기술 2 → rag_node.ask() 기술별 독립 호출.
출력 키: tech_summary · citations · retrieval_log · llm_calls

단독 실행: uv run python -m agents.tech_research   → outputs/tech_summary.json (LLM 30~50회)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from agents._common import TECHS, load_prompt
from graph.state import GraphState, TechSummary
from rag.rag_node import NO_EVIDENCE, ask

NODE = "tech_research"
LIST_FIELDS = ("numbers", "limitations", "apply_conditions")


def questions(tech: str) -> list[tuple[str, list[str]]]:
    """prompts/tech_research.md 의 `key: 질문 || 대체질문` 줄 → [(key, [질문, ...])]"""
    out = []
    for line in load_prompt("tech_research").splitlines():
        if ":" in line and not line.startswith("<!--") and not line.startswith("    "):
            key, q = line.split(":", 1)
            if key.strip() in TechSummary.__annotations__:
                out.append((key.strip(), [v.strip().replace("{tech}", tech) for v in q.split("||")]))
    return out


def _sentences(text: str) -> list[str]:
    """'문장[p.N]. 문장[p.N].' → 문장 리스트 (태그 유지). 근거 없음이면 빈 리스트."""
    if text.strip() == NO_EVIDENCE:
        return []
    # [p.N] 태그 뒤에서만 자른다 — 태그 없는 문장은 다음 태그 문장에 붙여 근거를 잃지 않게
    parts = re.split(r"(?<=\])\s*\.?\s+(?=\S)", text.strip())
    return [p.strip().rstrip(".") for p in parts if p.strip()]


def research(tech: str) -> tuple[TechSummary, list, list, int]:
    summary: dict = {k: [] for k in LIST_FIELDS} | {"overview": "", "mechanism": "", "evidence": []}
    citations, log, calls = [], [], 0
    for key, variants in questions(tech):
        for q in variants:                                # 대체 질문은 앞이 실패했을 때만
            r = ask(tech, q, node=NODE)
            calls += r["llm_calls"]
            log.append(r["retrieval_entry"])
            if r["answer"] != NO_EVIDENCE:
                break
        citations += r["citations"]
        summary["evidence"] += r["evidence"]
        if key in LIST_FIELDS:
            summary[key] = _sentences(r["answer"])
        else:
            summary[key] = r["answer"]
        if not r["faithful"]:
            summary["evidence"].append(
                {"claim": f"Faithfulness 미통과({key}): " + "; ".join(r["unsupported"]),
                 "tag": "추론", "ref": "judge", "page": None}
            )
    return summary, citations, log, calls


def run(state: GraphState) -> dict:
    tech_summary, citations, log, calls = {}, [], [], 0
    for tech in TECHS:                                   # 기술별 독립 호출 (장치 2)
        s, c, lg, n = research(tech)
        tech_summary[tech] = s
        citations += c
        log += lg
        calls += n
    return {"tech_summary": tech_summary, "citations": citations, "retrieval_log": log, "llm_calls": calls}


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(".env")
    out = run({})
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/tech_summary.json").write_text(
        json.dumps(out["tech_summary"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path("outputs/retrieval_log.json").write_text(
        json.dumps(out["retrieval_log"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    hits = sum(e["relevance"] == "yes" for e in out["retrieval_log"])
    print(f"llm_calls={out['llm_calls']}  retrieval yes={hits}/{len(out['retrieval_log'])}  → outputs/tech_summary.json")
    for tech, s in out["tech_summary"].items():
        print(f"\n[{tech}] overview: {s['overview'][:150]}")
        print(f"  numbers({len(s['numbers'])}): {s['numbers'][:2]}")
        print(f"  limitations({len(s['limitations'])}) apply({len(s['apply_conditions'])}) evidence({len(s['evidence'])})")
