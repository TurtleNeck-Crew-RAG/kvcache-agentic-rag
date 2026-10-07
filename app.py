"""실행 스크립트 — python app.py  [소유: D 황재원]

1. config/domain.yaml · config/selection.yaml 로드 → trace_id 생성 → init_state
2. 인덱스(data/index/)가 없으면 rag.indexing.build_index()  (--skip-index 로 건너뜀)
3. graph 를 stream 으로 실행 — observe.run_config: thread_id(체크포인터) · LangSmith metadata.trace_id · tags · run_name 이 같은 trace_id, recursion_limit = RECURSION_LIMIT
4. 실패해도 남는 것: outputs/run.json(trace_id · 어디까지 갔나 · step_count · retry · eval_attempts · llm_calls · node_status)
   + 채워진 키의 JSON (체크포인터의 마지막 State 기준)
5. 보고서: report_uri (B 이행 후) 또는 report_md → outputs/report/report.md (+ PDF, weasyprint 있을 때)

옵션: --skip-index  인덱싱 건너뜀 / --pdf-name <파일명>  제출용 PDF 이름
      --resume <trace_id>  체크포인트에서 이어서 실행 (SqliteSaver — langgraph-checkpoint-sqlite 가 있을 때만 프로세스를 넘어 재개)
      --timeout <초>       벽시계 상한 (기본 RUN_TIMEOUT). 넘으면 다음 노드 경계에서 멈추고 status=INTERRUPTED — --resume 으로 이어 간다
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from graph.build import build_graph
from graph.observe import new_trace_id, run_config
from graph.state import init_state
from graph.supervisor import MAX_STEPS, RECURSION_LIMIT

OUT = Path("outputs")
INDEX_DIR = Path("data/index")
CHECKPOINT_DB = OUT / "checkpoints.sqlite"
# 시간 상한은 게이트가 아니라 여기서 (#102) — 게이트가 시계를 읽으면 같은 State 에서 다른 결정이 나온다.
# 노드 경계에서만 확인하므로 LLM 호출 하나가 멈추는 경우는 호출 단위 timeout(agents/_common.py)이 1차 방어.
RUN_TIMEOUT = 1200        # 초 — ⚠️ 잠정. RAG 실행 186초 × 재작업 · 평가 루프 여유. 첫 실제 실행 소요 × 1.5 로 확정
DUMP_KEYS = ("citations", "synthesis", "trl_estimate", "tech_summary", "market_eval", "stakeholder_eval",
             "domain_eval", "sufficiency", "eval_result", "errors",
             "neutrality")                         # 이행 중 키 (graph/state.py)


def make_checkpointer():
    """(checkpointer, 프로세스를 넘어 재개 가능한가). SqliteSaver 가 없으면 InMemorySaver — 같은 프로세스 안에서만 남는다."""
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        from langgraph.checkpoint.memory import InMemorySaver
        return InMemorySaver(), False
    OUT.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)), True


def _dump(state: dict, visited: list[str], error: str | None, elapsed: float) -> None:
    """부분 State 라도 outputs/ 에 남긴다 — 어디서 깨졌는지 보기 위해."""
    OUT.mkdir(parents=True, exist_ok=True)
    for key in DUMP_KEYS:
        if state.get(key):                        # 빈 dict/list 는 파일을 만들지 않는다 (init_state 의 {} 와 구분)
            (OUT / f"{key}.json").write_text(json.dumps(state[key], ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "run.json").write_text(json.dumps({
        "ok": error is None,
        "error": error,
        "trace_id": state.get("trace_id"),
        "status": state.get("status"),
        "visited": visited,                       # 실행된 노드 순서 (supervisor 제외) — 재작업 · 평가 루프가 몇 번 돌았는지
        "step_count": state.get("step_count"),
        "llm_calls": state.get("llm_calls"),
        "tokens": state.get("tokens"),
        "retry": state.get("retry"),
        "eval_attempts": state.get("eval_attempts"),
        "node_status": state.get("node_status"),
        "last_error": state.get("last_error"),
        "report_uri": state.get("report_uri"),
        "elapsed_sec": round(elapsed, 1),
        "keys_filled": sorted(k for k in DUMP_KEYS if state.get(k)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_index(skip: bool) -> None:
    if skip or (INDEX_DIR.exists() and any(INDEX_DIR.iterdir())):
        return
    from rag.indexing import build_index  # A 소유 — 여기서만 import (없어도 그래프 조립은 되게)
    print("index: data/index/ 없음 → build_index()")
    build_index()


def _save_report(state: dict, visited: list[str]) -> None:
    from agents import report  # weasyprint 는 선택 의존성 — 여기서만 import
    if state.get("report_md"):                    # 이행 중 — report_md 를 파일로 (fallback · end_with_warning 이 고친 본문 포함)
        md = report.OUT_DIR / "report.md"
        if not md.exists() or md.read_text(encoding="utf-8") != state["report_md"]:
            report.save(state["report_md"])
    elif state.get("report_uri") and "end_with_warning" in visited:
        report.to_pdf(Path(state["report_uri"]))  # 자동 경고 절이 붙은 md 로 PDF 를 다시 만든다


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-index", action="store_true")
    ap.add_argument("--pdf-name", default=None, help="예: Agent-Output_판교_10반_박유진+황재원+민영은+심준용.pdf")
    ap.add_argument("--resume", metavar="TRACE_ID", default=None, help="체크포인트에서 이어서 실행")
    ap.add_argument("--timeout", type=float, default=RUN_TIMEOUT, help=f"벽시계 상한 초 (기본 {RUN_TIMEOUT})")
    args = ap.parse_args(argv)

    load_dotenv()
    if args.pdf_name:
        os.environ["REPORT_PDF_NAME"] = args.pdf_name   # 보고서 워커의 save() 가 이 이름으로 PDF 를 쓴다 (그래프 실행 전에)
    domain = yaml.safe_load(Path("config/domain.yaml").read_text(encoding="utf-8"))
    selected = yaml.safe_load(Path("config/selection.yaml").read_text(encoding="utf-8"))
    _ensure_index(args.skip_index)

    checkpointer, durable = make_checkpointer()
    if args.resume and not durable:
        print("--resume 은 SqliteSaver 가 필요합니다 (uv add langgraph-checkpoint-sqlite)", file=sys.stderr)
        return 2
    app = build_graph(checkpointer=checkpointer)
    trace_id = args.resume or new_trace_id()
    config = run_config(trace_id, RECURSION_LIMIT)   # thread_id = metadata.trace_id = tags = run_name (graph/observe.py)
    inputs = None if args.resume else init_state(domain, selected, trace_id=trace_id, max_steps=MAX_STEPS)
    print(f"trace_id={trace_id}" + ("  (resume)" if args.resume else ""))

    visited: list[str] = []
    error: str | None = None
    t0 = time.time()
    try:
        for chunk in app.stream(inputs, config=config, stream_mode="updates"):
            visited.extend(n for n in chunk if n != "supervisor")
            if time.time() - t0 > args.timeout:  # 체크포인트는 이 노드까지 저장돼 있다
                error = f"Timeout: {args.timeout:.0f}초 초과"
                print(f"graph: 시간 상한 — python app.py --resume {trace_id}", file=sys.stderr)
                break
    except KeyboardInterrupt:
        error = "KeyboardInterrupt"
        print(f"graph: 중단 — python app.py --resume {trace_id}", file=sys.stderr)
    except Exception as e:                        # noqa: BLE001 — 어디서 깨졌는지 남기는 게 목적
        error = f"{type(e).__name__}: {e}"
        print(f"graph: 실패 — {error}  (재개: python app.py --resume {trace_id})", file=sys.stderr)
    state = dict(app.get_state(config).values)    # 체크포인터의 마지막 State — 중간에 죽어도 남는다
    if error:
        state["status"] = "INTERRUPTED" if error == "KeyboardInterrupt" or error.startswith("Timeout") else "FAILED"
    _dump(state, visited, error, time.time() - t0)

    if state.get("report_md") or state.get("report_uri"):
        _save_report(state, visited)
    else:
        print("보고서 없음 — outputs/run.json 의 visited · error 확인", file=sys.stderr)

    print(f"trace_id={trace_id}  status={state.get('status')}  step_count={state.get('step_count')}  "
          f"llm_calls={state.get('llm_calls')}  tokens={state.get('tokens')}  retry={state.get('retry')}  eval_attempts={state.get('eval_attempts')}"
          "  → outputs/run.json")
    return 0 if error is None and state.get("status") == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
