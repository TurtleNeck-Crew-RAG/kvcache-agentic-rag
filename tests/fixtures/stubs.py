"""손으로 쓴 fixtures 로 도는 가짜 워커 · 판정 노드 — 그래프 통합 테스트용 (LLM·네트워크 없음).  [소유: D 황재원]

워커 계약(ROLES.md 4절)을 그대로 따른다: 자기 출력 키 + citations + llm_calls 만, rework_request 가 자기 것이면 그 기술만.
판정 stub 은 A assess · C evaluator 의 출력 형식(CellVerdict · EvalResult)만 맞춘다.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from graph.state import TECHS

FIX = Path(__file__).parent


def _load(name: str) -> dict:
    d = json.loads((FIX / name).read_text(encoding="utf-8"))
    d.pop("_note", None)
    return d


def _techs(state: dict, me: str) -> tuple[str, ...]:
    req = state.get("rework_request") or {}
    return (req["tech"],) if req.get("worker") == me else TECHS


def make_workers(calls: list[str] | None = None, fail: set[str] = frozenset()) -> dict[str, Callable]:
    """calls 에 실행 순서를 남긴다 (재작업이면 "market:InfiniGen"). fail 에 든 워커는 예외."""
    calls = [] if calls is None else calls
    summary, evals, synth = _load("tech_summary.json"), _load("evals.json"), _load("synthesis.json")
    payload = {"tech_research": ("tech_summary", summary), "market": ("market_eval", evals["market_eval"]),
               "stakeholder": ("stakeholder_eval", evals["stakeholder_eval"]), "domain": ("domain_eval", evals["domain_eval"])}

    def cell_worker(name: str):
        key, data = payload[name]

        def run(state):
            techs = _techs(state, name)
            calls.append(name if len(techs) == len(TECHS) else f"{name}:{techs[0]}")
            if name in fail:
                raise RuntimeError(f"{name} stub 실패")
            return {key: {t: data[t] for t in techs}, "llm_calls": len(techs)}
        return run

    def synthesis(state):
        calls.append("synthesis")
        return {"synthesis": synth["synthesis"], "trl_estimate": synth["trl_estimate"],
                "citations": synth["citations"], "llm_calls": 1}

    def report(state):
        calls.append("report")
        fb = (state.get("eval_result") or {}).get("feedback", "")
        return {"report_md": f"# stub report\n\n{fb}\n", "llm_calls": 1}

    return {**{n: cell_worker(n) for n in payload}, "synthesis": synthesis, "report": report}


def make_assess(insufficient: dict[str, int] | None = None) -> Callable:
    """insufficient = {"market:InfiniGen": k} — 그 셀을 처음 k 번 판정에서 부족으로 본다. 실패 노드 셀은 항상 부족 (A 규칙)."""
    seen: dict[str, int] = {}
    insufficient = insufficient or {}

    def assess(state):
        """판정이 없는 셀 + 방금 재작업한 셀만 판정한다 (이미 판정된 셀을 매 턴 다시 Judge 에 보내지 않는다)."""
        out = {}
        verdicts = state.get("sufficiency") or {}
        req = state.get("rework_request") or {}
        for w, key in (("tech_research", "tech_summary"), ("market", "market_eval"),
                       ("stakeholder", "stakeholder_eval"), ("domain", "domain_eval")):
            for t in (state.get(key) or {}):
                cell = f"{w}:{t}"
                if cell in verdicts and not (req.get("worker") == w and req.get("tech") == t):
                    continue
                seen[cell] = seen.get(cell, 0) + 1
                bad = seen[cell] <= insufficient.get(cell, 0) or (state.get("node_status") or {}).get(w) == "failed"
                out[cell] = {"rule": "fail" if bad else "pass", "judge": None if bad else "sufficient",
                             "gap": "negatives" if bad else "", "hint_query": f"{t} limitations" if bad else "",
                             "reason": "stub"}
        return {"sufficiency": out}
    return assess


def make_evaluator(fail_times: int = 0, targets: list[str] | None = None) -> Callable:
    """처음 fail_times 번은 fail (targets 로 되돌림), 그 뒤 pass."""
    n = {"i": 0}
    ok = {"passed": True, "score": 1.0, "reason": ""}

    def evaluator(state):
        n["i"] += 1
        passed = n["i"] > fail_times
        g = ok if passed else {"passed": False, "score": 0.4, "reason": "stub"}
        return {"eval_result": {"passed": passed, "items": {"groundedness": g, "neutrality": ok, "bias": ok, "coverage": ok},
                                "targets": [] if passed else list(targets or ["report"]),
                                "feedback": "" if passed else "근거 보강"}}
    return evaluator
