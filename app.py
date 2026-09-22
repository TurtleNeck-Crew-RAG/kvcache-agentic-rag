"""실행 스크립트 — python app.py

1. config/domain.yaml · config/selection.yaml 로드 → init_state
2. 인덱스(data/index/)가 없으면 rag.indexing.build_index()  (--skip-index 로 건너뜀)
3. graph 를 stream 으로 실행 (recursion_limit 40 — 안전망) — 노드가 끝날 때마다 마지막 State 를 붙잡아 둔다
4. 실패해도 남는 것: outputs/run.json(어디까지 갔나 · llm_calls · retry) + retrieval_log · citations · synthesis · trl_estimate JSON
5. report_md 가 있으면 outputs/report/report.md (+ PDF, weasyprint 있을 때)

옵션: --skip-index  인덱싱 건너뜀 / --pdf-name <파일명>  제출용 PDF 이름
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
from graph.state import init_state

OUT = Path("outputs")
INDEX_DIR = Path("data/index")
DUMP_KEYS = ("retrieval_log", "citations", "synthesis", "trl_estimate", "tech_summary",
             "market_eval", "stakeholder_eval", "domain_eval", "neutrality")


def _dump(state: dict, visited: list[str], error: str | None, elapsed: float) -> None:
    """부분 State 라도 outputs/ 에 남긴다 — 어디서 깨졌는지 보기 위해."""
    OUT.mkdir(parents=True, exist_ok=True)
    for key in DUMP_KEYS:
        if state.get(key):                        # 빈 dict/list 는 파일을 만들지 않는다 (init_state 의 {} 와 구분)
            (OUT / f"{key}.json").write_text(json.dumps(state[key], ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "run.json").write_text(json.dumps({
        "ok": error is None,
        "error": error,
        "visited": visited,                       # Dispatcher 결정 순서 — 규칙표대로 갔는지 확인 (병렬은 " | " 로 묶임)
        "llm_calls": state.get("llm_calls"),
        "retry": state.get("retry"),
        "elapsed_sec": round(elapsed, 1),
        "keys_filled": sorted(k for k in DUMP_KEYS if state.get(k)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_index(skip: bool) -> None:
    if skip or (INDEX_DIR.exists() and any(INDEX_DIR.iterdir())):
        return
    from rag.indexing import build_index  # 여기서만 import (없어도 그래프 조립은 되게)
    print("index: data/index/ 없음 → build_index()")
    build_index()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-index", action="store_true")
    ap.add_argument("--pdf-name", default=None, help="예: RAG-Output_판교_10반_박유진+황재원+민영은+심준용.pdf")
    args = ap.parse_args(argv)

    load_dotenv()
    if args.pdf_name:
        os.environ["REPORT_PDF_NAME"] = args.pdf_name   # 보고서 워커의 save() 가 이 이름으로 PDF 를 쓴다 (그래프 실행 전에)
    domain = yaml.safe_load(Path("config/domain.yaml").read_text(encoding="utf-8"))
    selected = yaml.safe_load(Path("config/selection.yaml").read_text(encoding="utf-8"))
    _ensure_index(args.skip_index)

    app = build_graph()
    state: dict = dict(init_state(domain, selected))
    visited: list[str] = []
    error: str | None = None
    t0 = time.time()
    try:
        # stream_mode="values" — 슈퍼스텝마다 전체 State 가 온다. 중간에 죽어도 마지막 것을 갖는다
        for step in app.stream(state, config={"recursion_limit": 40}, stream_mode="values"):
            state = step
            nxt = " | ".join(state.get("next") or [])   # Dispatcher 가 정한 다음 노드(들). 병렬이면 "market | stakeholder | domain"
            if nxt and (not visited or visited[-1] != nxt):
                visited.append(nxt)
    except Exception as e:                        # noqa: BLE001 — 어디서 깨졌는지 남기는 게 목적
        error = f"{type(e).__name__}: {e}"
        print(f"graph: 실패 — {error}", file=sys.stderr)
    _dump(state, visited, error, time.time() - t0)

    if state.get("report_md"):
        from agents import report  # weasyprint 는 선택 의존성 — 여기서만 import
        md = report.OUT_DIR / "report.md"
        if not md.exists() or md.read_text(encoding="utf-8") != state["report_md"]:
            report.save(state["report_md"])       # 워커가 실패 기록(fallback)이라 저장을 못 했을 때만
    else:
        print("report_md 없음 — outputs/run.json 의 visited · error 확인", file=sys.stderr)

    print(f"llm_calls={state.get('llm_calls')}  retry={state.get('retry')}  visited={len(visited)}  → outputs/run.json")
    return 0 if error is None and state.get("report_md") else 1


if __name__ == "__main__":
    sys.exit(main())
