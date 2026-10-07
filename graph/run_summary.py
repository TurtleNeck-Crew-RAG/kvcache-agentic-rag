"""실행 요약 — 한 번의 실행(trace_id)이 어떻게 흘렀는지 한 장으로.  [소유: A 박유진]

app.py 가 남긴 outputs/run.json · sufficiency.json · eval_result.json 과 observe 의 decisions.jsonl 을 읽어
라우팅 순서 · 셀별 재작업 · 끝까지 부족한 셀 · 품질 평가를 마크다운으로 정리한다.

쓰임
- 첫 실제 실행 뒤 충분성 기준이 엉뚱하게 걸린 셀이 있는지 확인 (ROLES.md 2절 A — 오판정이 있을 때만 기준 조정)
- LangSmith 에서 캡처할 실행 고르기 (재작업 · 평가 루프가 찍힌 실행)
- README Run Record 표 (동적 동작 실증)

    uv run python -m graph.run_summary                 # outputs/run.json 의 trace_id
    uv run python -m graph.run_summary <trace_id> --md # outputs/run_summary.md 로도 저장
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from graph import observe


def _load(name: str) -> Any:
    path = observe.OUT / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def summarize(trace_id: str | None = None) -> dict[str, Any]:
    run = _load("run.json") or {}
    trace_id = trace_id or run.get("trace_id") or ""
    same_run = run.get("trace_id") == trace_id          # run.json · *.json 은 마지막 실행 것뿐
    decisions = [d for d in observe.read_jsonl(observe.DECISIONS, trace_id) if d.get("node") == "supervisor"]
    sufficiency = (_load("sufficiency.json") if same_run else None) or {}
    eval_result = _load("eval_result.json") if same_run else None
    retry = (run.get("retry") if same_run else None) or {}

    reworks = Counter()
    for d in decisions:
        if "재작업" in d.get("reason", "") and "부족 셀" in d.get("reason", ""):
            reworks[d["reason"].split("부족 셀 ", 1)[1].split()[0]] += 1
    eval_loops = sum("평가 fail" in d.get("reason", "") and "루프" in d.get("reason", "") for d in decisions)
    unresolved = {c: v for c, v in sufficiency.items()
                  if v.get("rule") == "fail" or v.get("judge") == "insufficient"}
    return {
        "trace_id": trace_id,
        "run": run if same_run else {},
        "decisions": decisions,
        "routing_count": len(decisions),
        "reworks": dict(reworks) or retry,
        "rework_total": sum(reworks.values()) or sum(retry.values()),
        "eval_loops": eval_loops,
        "sufficiency": sufficiency,
        "unresolved": unresolved,
        "eval_result": eval_result,
    }


def to_markdown(s: dict[str, Any]) -> str:
    run = s["run"]
    out = [f"# 실행 요약 — `{s['trace_id']}`", ""]
    if run:
        out += ["| 항목 | 값 |", "|---|---|",
                f"| status | {run.get('status')} |",
                f"| Supervisor 라우팅 | {s['routing_count']}회 (step_count {run.get('step_count')}) |",
                f"| 셀 재작업 | {s['rework_total']}회 — {', '.join(f'{c} ×{n}' for c, n in s['reworks'].items()) or '없음'} |",
                f"| 평가 루프 | {s['eval_loops']}회 (eval_attempts {run.get('eval_attempts')}) |",
                f"| LLM 호출 | {run.get('llm_calls')} |",
                f"| 소요 | {run.get('elapsed_sec')}초 |", ""]
    else:
        out += ["(run.json 이 다른 실행 것 — 결정 로그만 표시)", ""]

    out += ["## 라우팅 순서 (Supervisor 결정)", "", "| step | → | 사유 |", "|---|---|---|"]
    out += [f"| {d.get('step')} | `{d.get('decision')}` | {d.get('reason', '').replace('|', '/')} |" for d in s["decisions"]]

    if s["sufficiency"]:
        out += ["", "## 셀별 충분성 (마지막 판정)", "", "| 셀 | 규칙 | Judge | gap | 재작업 | 사유 |", "|---|---|---|---|---|---|"]
        for cell, v in sorted(s["sufficiency"].items()):
            out.append(f"| {cell} | {v.get('rule')} | {v.get('judge') or '—'} | {v.get('gap') or '—'} | "
                       f"{s['reworks'].get(cell, 0)} | {v.get('reason', '').replace('|', '/')[:120]} |")
        if s["unresolved"]:
            out += ["", "**끝까지 부족한 셀** (보고서 한계점 대상): " + ", ".join(sorted(s["unresolved"]))]

    ev = s["eval_result"]
    if ev:
        out += ["", f"## 품질 평가 — {'pass' if ev.get('passed') else 'fail'}", "", "| 항목 | 통과 | 점수 | 사유 |", "|---|---|---|---|"]
        for k, it in (ev.get("items") or {}).items():
            out.append(f"| {k} | {it.get('passed')} | {it.get('score')} | {str(it.get('reason', '')).replace('|', '/')[:120]} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_id", nargs="?", default=None)
    ap.add_argument("--md", action="store_true", help="outputs/run_summary.md 로 저장")
    args = ap.parse_args(argv)
    md = to_markdown(summarize(args.trace_id))
    print(md)
    if args.md:
        (observe.OUT / "run_summary.md").write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
