"""Supervisor — 결정론 게이트 (docs/ROLES.md 0절 · 2절 D).  [소유: D 황재원]

판정은 확률(assess · evaluator, LLM Judge), 게이트는 결정론(여기, LLM 없음) — 교안 부록 B.
supervisor(state) 는 판정 결과(sufficiency · eval_result) · 시도 횟수(retry · eval_attempts) · 상한만 읽고
매 턴 next 1개를 정한다. route(state) 는 state["next"] 를 돌려줄 뿐이다.

우선순위 — 순서가 고정된 파이프라인이 아니라 State 의 빈칸 · 부족 셀 · 평가 결과에서 매 턴 계산한다
  0. step_count ≥ max_steps                       → end_with_warning
  1·2는 선행 조건 단계(STAGES)별로 — tech_summary 가 나머지 워커의 입력이라 기술 조사 셀이 충분/소진돼야 다음 단계로
  1. 미수집 셀                                    → 해당 워커
     수집됐는데 충분성 판정이 없는 셀 (assess 실패)   → end_with_warning — 판정 없음을 "충분"으로 보지 않는다 (#120)
  2. 부족 셀 (재작업 < MAX_REWORK, 예산 안)          → 해당 워커 + rework_request
     라운드 로빈 — 재작업 횟수가 가장 적은 셀부터. 모든 부족 셀이 1회씩 받은 뒤에야 2회째 (뒤쪽 관점이 예산에 밀려 잘리지 않게)
  3. synthesis 없음                              → synthesis
  4. 보고서 없음                                  → report   (→ evaluator 는 엣지로 고정)
  5. 평가 pass                                   → END  — 단 node_status 에 failed 가 남아 있으면 end_with_warning
     보고서는 있는데 eval_result 가 없음(evaluator 실패) → end_with_warning — 평가 없이 성공으로 끝내지 않는다 (#120)
  6. 평가 fail (eval_attempts < MAX_EVAL, 예산 안)  → targets 중 첫 실행 가능한 곳 — 재조사면 synthesis · 보고서를 비워 다시 흐르게
  7. 평가 fail 소진                              → end_with_warning

예산(#102) — 재작업 · 평가 루프는 "마무리(종합 · 보고서 · 평가) 예약분을 남기고도 예산 안"일 때만.
예산이 바닥나도 보고서는 끝까지 만든다. 단위는 토큰(safe.py 가 노드마다 집계). 호출 수(llm_calls)는 보고용.
시간 제한은 게이트가 아니라 app.py 의 RUN_TIMEOUT — 게이트가 시계를 읽으면 같은 State 에서 다른 결정이 나온다.

결정 사유는 observe.log_decision() 으로 State 밖에 남긴다 (A 소유 — 없으면 아래 fallback).
재작업 사유 형식 "부족 셀 {cell} 재작업 k/MAX — gap={gap} · {판정 사유}" 는 graph/run_summary.py 가 파싱한다 — 바꾸면 같이 고칠 것.
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


WORKERS = ("tech_research", "market", "stakeholder", "domain")     # 셀을 가진 워커 — 라운드 로빈 동률이면 이 순서
STAGES = (("tech_research",), ("market", "stakeholder", "domain"))  # 선행 조건 — 앞 단계 셀이 충분/소진돼야 다음 단계

# ── 상한 — README State Schema 「종료 보장」에 그대로 적는다 ──
MAX_REWORK = 2            # 셀("{worker}:{tech}")별 재작업 — 평가 루프의 관점 재조사도 같은 카운터
MAX_EVAL = 2              # 보고서 평가 fail 후 되돌리는 횟수
# Supervisor 턴 상한은 구조에서 계산한다 (실측값이 아니라 "구조상 최대 + 1" 안전망)
CELLS = len(WORKERS) * len(TECHS)                                   # 8
BASE_TURNS = len(WORKERS) + 3                                       # 수집 4 + synthesis + report + END = 7
MAX_STEPS = BASE_TURNS + CELLS * MAX_REWORK + MAX_EVAL * 3 + 1      # 7 + 16 + 6 + 1 = 30 (평가 루프 1회 ≤ 재조사 · 종합 · 보고서)
RECURSION_LIMIT = 3 * MAX_STEPS + 10                                # 턴당 노드 ≤3 (supervisor → 워커 → assess/evaluator)

# 예산 (#102) — 실측 2회(20261007-163346-6f027b · 20261007-164001-e33f18) 기준. 두 실행 모두 재작업 · 평가 루프를
# 다 쓴 최악에 가까운 경로였다 (토큰 381k · 397k, LLM 호출 132 · 141).
TOKEN_BUDGET = 600_000    # 최대 실측 397k × 1.5 ≈ 596k
FINAL_RESERVE = 80_000    # synthesis 19k + report 21k + evaluator 2k ≈ 42k (노드별 실측) 의 약 2배 — 넘으면 재작업 · 평가 루프를 멈추고 마무리
# 호출 수 상한(LLM_BUDGET 150, #29)은 제거 — 토큰이 비용 단위가 됐고, 실측 141회로 표본 편차만으로 토큰보다 먼저 걸렸다
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


def _budget(state: GraphState) -> tuple[bool, str]:
    """재작업 · 평가 루프를 더 허용하는가 (허용, 막힌 사유). 마무리 예약분을 남겨 둔다."""
    tokens = state.get("tokens", 0)
    if tokens + FINAL_RESERVE > TOKEN_BUDGET:
        return False, f"tokens {tokens} + FINAL_RESERVE {FINAL_RESERVE} > TOKEN_BUDGET {TOKEN_BUDGET}"
    return True, ""


def _decide(state: GraphState) -> tuple[str, str, dict[str, Any]]:
    """(next, 사유, 추가 갱신). 순수 함수 — State 를 읽기만 한다."""
    steps, max_steps = state.get("step_count", 0), state.get("max_steps", MAX_STEPS)
    if steps >= max_steps:
        return "end_with_warning", f"step_count {steps} ≥ max_steps {max_steps}", {}

    retry = state.get("retry") or {}
    budget_ok, budget_why = _budget(state)
    verdicts = state.get("sufficiency") or {}
    for stage in STAGES:
        for w in stage:                                                       # 1
            missing = [t for t in TECHS if t not in (state.get(PAYLOAD[w]) or {})]
            if missing:
                return w, f"미수집 셀 {', '.join(f'{w}:{t}' for t in missing)}", {}
        unjudged = [f"{w}:{t}" for w in stage for t in TECHS if f"{w}:{t}" not in verdicts]
        if unjudged:                                                          # assess 실패 — 평가 없이 다음 단계로 가지 않는다
            return "end_with_warning", f"충분성 미판정 셀 {', '.join(unjudged)}", {}
        if not budget_ok:
            continue                                                          # 예산 소진 — 부족 셀은 한계점에 남는다
        candidates = [                                                        # 2 — 라운드 로빈: (재작업 횟수, 워커 순, 기술 순)
            (retry.get(f"{w}:{t}", 0), WORKERS.index(w), TECHS.index(t), w, t)
            for w in stage for t in TECHS
            if (v := verdicts.get(f"{w}:{t}")) and _insufficient(v) and retry.get(f"{w}:{t}", 0) < MAX_REWORK
        ]
        if candidates:
            n, _, _, w, t = min(candidates)
            cell, v = f"{w}:{t}", verdicts[f"{w}:{t}"]
            req = ReworkRequest(worker=w, tech=t, gap=v.get("gap", ""), hint_query=v.get("hint_query", ""))
            return w, f"부족 셀 {cell} 재작업 {n + 1}/{MAX_REWORK} — gap={v.get('gap', '')} · {v.get('reason', '')}", {
                "rework_request": req, "retry": {cell: n + 1}}

    if not state.get("synthesis"):                                            # 3
        return "synthesis", "수집 셀 전부 충분 또는 재작업 소진", {}
    if not _has_report(state):                                                # 4
        return "report", "종합 완료 · 보고서 없음", {}

    ev = state.get("eval_result")
    if ev is None:                                                            # 5 — evaluator 실패 · 미실행
        return "end_with_warning", "품질 평가 결과 없음 — 평가 없이 성공으로 끝내지 않는다", {}
    if ev.get("passed"):
        failed_nodes = sorted(n for n, st in (state.get("node_status") or {}).items() if st == "failed")
        if failed_nodes:                                                      # safe 의 fallback 을 성공으로 끝내지 않는다
            return "end_with_warning", f"실패 노드 잔존 {', '.join(failed_nodes)}", {}
        return END, "보고서 평가 pass", {"status": "SUCCESS"}

    attempts = state.get("eval_attempts", 0)                                  # 6 · 7
    failed = ", ".join(k for k, it in (ev.get("items") or {}).items() if not it.get("passed"))
    if attempts >= MAX_EVAL or not budget_ok:
        why = f"eval_attempts {attempts} ≥ {MAX_EVAL}" if budget_ok else budget_why
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
    budget_ok, budget_why = _budget(state)
    if not budget_ok:
        out.append(f"예산 소진 ({budget_why}) — 이후 재작업 · 평가 루프 중단, 마무리 예약분으로 보고서")
    retry = state.get("retry") or {}
    verdicts = state.get("sufficiency") or {}
    for cell, v in verdicts.items():
        if _insufficient(v):
            out.append(f"근거 부족 셀 {cell} (재작업 {retry.get(cell, 0)}/{MAX_REWORK}) — {v.get('reason') or v.get('gap', '')}")
    for w in WORKERS:
        for t in TECHS:
            if t in (state.get(PAYLOAD[w]) or {}) and f"{w}:{t}" not in verdicts:
                out.append(f"충분성 미평가 셀 {w}:{t} — 판정 노드(assess) 실패로 근거 충분성을 평가하지 못함")
    for node, st in (state.get("node_status") or {}).items():
        if st == "failed":
            out.append(f"노드 실패 {node}")
    ev = state.get("eval_result")
    if _has_report(state) and ev is None:
        out.append("품질 평가 미실행 — 평가 노드(evaluator) 실패로 보고서 품질을 평가하지 못함")
    if ev and not ev.get("passed"):
        for k, it in (ev.get("items") or {}).items():
            if not it.get("passed"):
                out.append(f"품질 평가 미달 {k} (score {it.get('score')}) — {it.get('reason', '')}")
    return out


REFERENCE_HEADING = "\n## REFERENCE"


def _warning_md(items: list[str]) -> str:
    lines = "\n".join(f"- {x}" for x in items) or "- (기록된 미달 항목 없음)"
    return f"## 자동 경고 — 상한 소진으로 종료\n\n{lines}\n\n"


def _insert_warning(md: str, items: list[str]) -> str:
    """경고 절을 REFERENCE 앞(6장 한계점 뒤)에 넣는다 — 참고문헌 뒤에 본문이 오지 않게 (#121). REFERENCE 가 없으면 끝에."""
    block = _warning_md(items)
    i = md.find(REFERENCE_HEADING)
    if i < 0:
        return md.rstrip("\n") + "\n\n" + block
    return md[:i].rstrip("\n") + "\n\n" + block + md[i + 1:]


def end_with_warning(state: GraphState) -> dict:
    items = unmet(state)
    log_decision(state, "end_with_warning", END, "; ".join(items))
    out: dict[str, Any] = {"status": "SUCCESS" if _has_report(state) else "INTERRUPTED"}
    uri = state.get("report_uri")
    if uri and Path(uri).exists():
        path = Path(uri)
        path.write_text(_insert_warning(path.read_text(encoding="utf-8"), items), encoding="utf-8")
    elif state.get("report_md"):                         # safe fallback 의 report_md — app.py 가 파일 · PDF 로 저장
        out["report_md"] = _insert_warning(state["report_md"], items)
    return out
