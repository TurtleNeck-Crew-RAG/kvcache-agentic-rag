"""Supervisor — 결정론 게이트 (docs/ROLES.md 0절 · 2절 D).  [소유: D 황재원]

판정은 확률(assess · evaluator, LLM Judge), 게이트는 결정론(여기, LLM 없음) — 교안 부록 B.
supervisor(state) 는 판정 결과(sufficiency · eval_result) · 시도 횟수(retry · eval_attempts) · 상한만 읽고
매 턴 next 1개를 정한다. route(state) 는 state["next"] 를 돌려줄 뿐이다.

우선순위 — 순서가 고정된 파이프라인이 아니라 State 의 빈칸 · 부족 셀 · 평가 결과에서 매 턴 계산한다
  0. step_count ≥ max_steps                       → end_with_warning
  1·2는 선행 조건 단계(STAGES)별로 — tech_summary 가 나머지 워커의 입력이라 기술 조사 셀이 충분/소진돼야 다음 단계로
  1. 미수집 셀                                    → 해당 워커
  2. 부족 셀 (재작업 < MAX_REWORK, 예산 안)          → 해당 워커 + rework_request
  3. synthesis 없음                              → synthesis
  4. 보고서 없음                                  → report   (→ evaluator 는 엣지로 고정)
  5. 평가 pass (또는 평가 노드 미연결)               → END
  6. 평가 fail (eval_attempts < MAX_EVAL, 예산 안)  → targets 중 첫 실행 가능한 곳 — 재조사면 synthesis · 보고서를 비워 다시 흐르게
  7. 평가 fail 소진                              → end_with_warning

결정 사유는 observe.log_decision() 으로 State 밖에 남긴다 (A 소유 — 없으면 아래 fallback).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END

from graph.state import TECHS, GraphState, ReworkRequest

try:
    from graph.observe import log_decision
except ImportError:                                   # A 의 observe 머지 전 — 같은 시그니처
    def log_decision(state: dict, node: str, decision: str, reason: str) -> None:
        print(f"[{node}] {state.get('trace_id', '')} → {decision}: {reason}", file=sys.stderr)


# ── 상한 — README State Schema 「종료 보장」에 그대로 적는다 ──
MAX_STEPS = 30            # supervisor 턴 수. 정상 7 + 셀 재작업 최대 8×2 + 평가 루프 2×3 = 29
MAX_REWORK = 2            # 셀("{worker}:{tech}")별 재작업
MAX_EVAL = 2              # 보고서 평가 fail 후 되돌리는 횟수
LLM_BUDGET = 150          # 넘으면 재작업 · 평가 루프만 멈추고 최단 경로로 보고서 (#29: 100 → 150)
RECURSION_LIMIT = 3 * MAX_STEPS + 10   # 턴당 노드 ≤3 (supervisor → 워커 → assess/evaluator)

WORKERS = ("tech_research", "market", "stakeholder", "domain")     # 셀을 가진 워커 — 같은 단계 안에서는 이 순서가 tie-break
STAGES = (("tech_research",), ("market", "stakeholder", "domain"))  # 선행 조건 — 앞 단계 셀이 충분/소진돼야 다음 단계
PAYLOAD = {"tech_research": "tech_summary", "market": "market_eval",
           "stakeholder": "stakeholder_eval", "domain": "domain_eval"}

Next = Literal["tech_research", "market", "stakeholder", "domain", "synthesis", "report",
               "end_with_warning", "__end__"]
NEXT_NODES: tuple[str, ...] = (*WORKERS, "synthesis", "report", "end_with_warning", END)

CLEAR_REPORT = {"report_uri": None, "report_md": None}
CLEAR_SYNTHESIS = {"synthesis": None, **CLEAR_REPORT}


def _insufficient(v: dict) -> bool:
    return v.get("rule") == "fail" or v.get("judge") == "insufficient"


def _has_report(state: GraphState) -> bool:
    return bool(state.get("report_uri") or state.get("report_md"))


def _decide(state: GraphState) -> tuple[str, str, dict[str, Any]]:
    """(next, 사유, 추가 갱신). 순수 함수 — State 를 읽기만 한다."""
    steps, max_steps = state.get("step_count", 0), state.get("max_steps", MAX_STEPS)
    if steps >= max_steps:
        return "end_with_warning", f"step_count {steps} ≥ max_steps {max_steps}", {}

    retry = state.get("retry") or {}
    calls = state.get("llm_calls", 0)
    budget_ok = calls <= LLM_BUDGET
    verdicts = state.get("sufficiency") or {}
    for stage in STAGES:
        for w in stage:                                                       # 1
            missing = [t for t in TECHS if t not in (state.get(PAYLOAD[w]) or {})]
            if missing:
                return w, f"미수집 셀 {', '.join(f'{w}:{t}' for t in missing)}", {}
        for w in stage:                                                       # 2
            for t in TECHS:
                cell = f"{w}:{t}"
                v = verdicts.get(cell)
                if not v or not _insufficient(v):
                    continue
                n = retry.get(cell, 0)
                if n >= MAX_REWORK or not budget_ok:
                    continue                                                  # 소진 — end_with_warning / 보고서 한계점에 남는다
                req = ReworkRequest(worker=w, tech=t, gap=v.get("gap", ""), hint_query=v.get("hint_query", ""))
                return w, f"부족 셀 {cell} 재작업 {n + 1}/{MAX_REWORK} — {v.get('reason', '')}", {
                    "rework_request": req, "retry": {cell: n + 1}}

    if not state.get("synthesis"):                                            # 3
        return "synthesis", "수집 셀 전부 충분 또는 재작업 소진", {}
    if not _has_report(state):                                                # 4
        return "report", "종합 완료 · 보고서 없음", {}

    ev = state.get("eval_result")
    if ev is None:                                                            # 5 — evaluator 머지 전
        return END, "eval_result 없음 — 평가 노드 미연결", {"status": "SUCCESS"}
    if ev.get("passed"):
        return END, "보고서 평가 pass", {"status": "SUCCESS"}

    attempts = state.get("eval_attempts", 0)                                  # 6 · 7
    failed = ", ".join(k for k, it in (ev.get("items") or {}).items() if not it.get("passed"))
    if attempts >= MAX_EVAL or not budget_ok:
        why = f"eval_attempts {attempts} ≥ {MAX_EVAL}" if budget_ok else f"llm_calls {calls} > LLM_BUDGET {LLM_BUDGET}"
        return "end_with_warning", f"보고서 평가 fail({failed}) · {why}", {}

    base = {"eval_attempts": attempts + 1}
    tag = f"평가 fail({failed}) 루프 {attempts + 1}/{MAX_EVAL}"
    for target in ev.get("targets") or []:
        if target == "report":
            return "report", f"{tag} → 보고서 재작성", {**base, **CLEAR_REPORT}
        if target == "synthesis":
            return "synthesis", f"{tag} → 종합 재실행", {**base, **CLEAR_SYNTHESIS}
        w, _, t = target.partition(":")
        if w in PAYLOAD and t in TECHS and retry.get(target, 0) < MAX_REWORK:
            req = ReworkRequest(worker=w, tech=t, gap="eval", hint_query=ev.get("feedback", ""))
            return w, f"{tag} → {target} 재조사", {
                **base, **CLEAR_SYNTHESIS, "rework_request": req, "retry": {target: retry.get(target, 0) + 1}}
    return "report", f"{tag} → 실행 가능한 target 없음, 보고서 재작성", {**base, **CLEAR_REPORT}


def supervisor(state: GraphState) -> dict:
    """노드. next 1개 + rework_request + step_count += 1 (+ retry · eval_attempts · 비울 키)."""
    nxt, reason, extra = _decide(state)
    log_decision(state, "supervisor", nxt, reason)
    return {"next": nxt, "rework_request": None, "step_count": 1, **extra}


def route(state: GraphState) -> Next:
    """조건부 엣지 — state["next"] 만 반환. 목적지 목록은 NEXT_NODES."""
    return state["next"]  # type: ignore[return-value]


# ── end_with_warning — 상한 소진 시 FAILED 로 죽지 않고 미달 항목을 남기고 끝낸다 ──

def unmet(state: GraphState) -> list[str]:
    """보고서 한계점에 남길 미달 항목."""
    out: list[str] = []
    steps, max_steps = state.get("step_count", 0), state.get("max_steps", MAX_STEPS)
    if steps >= max_steps:
        out.append(f"Supervisor 턴 상한 도달 (step_count {steps} / max_steps {max_steps})")
    if state.get("llm_calls", 0) > LLM_BUDGET:
        out.append(f"LLM 호출 예산 초과 (llm_calls {state.get('llm_calls')} / {LLM_BUDGET}) — 이후 재작업 · 평가 루프 중단")
    retry = state.get("retry") or {}
    for cell, v in (state.get("sufficiency") or {}).items():
        if _insufficient(v):
            out.append(f"근거 부족 셀 {cell} (재작업 {retry.get(cell, 0)}/{MAX_REWORK}) — {v.get('reason') or v.get('gap', '')}")
    for node, st in (state.get("node_status") or {}).items():
        if st == "failed":
            out.append(f"노드 실패 {node}")
    ev = state.get("eval_result")
    if ev and not ev.get("passed"):
        for k, it in (ev.get("items") or {}).items():
            if not it.get("passed"):
                out.append(f"품질 평가 미달 {k} (score {it.get('score')}) — {it.get('reason', '')}")
    return out


def _warning_md(items: list[str]) -> str:
    lines = "\n".join(f"- {x}" for x in items) or "- (기록된 미달 항목 없음)"
    return f"\n\n## 자동 경고 — 상한 소진으로 종료\n\n{lines}\n"


def end_with_warning(state: GraphState) -> dict:
    items = unmet(state)
    log_decision(state, "end_with_warning", END, "; ".join(items))
    out: dict[str, Any] = {"status": "SUCCESS" if _has_report(state) else "INTERRUPTED"}
    uri = state.get("report_uri")
    if uri and Path(uri).exists():
        with open(uri, "a", encoding="utf-8") as f:
            f.write(_warning_md(items))
    elif state.get("report_md"):                         # 이행 중 — app.py 가 report_md 를 파일 · PDF 로 저장
        out["report_md"] = state["report_md"] + _warning_md(items)
    return out
