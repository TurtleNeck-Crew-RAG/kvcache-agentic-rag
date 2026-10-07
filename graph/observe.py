"""관측성 — trace_id · 결정 로그 · 검색 로그를 State 밖에 남긴다.  [소유: A 박유진]

State 에는 상관 키 trace_id 하나만 두고, 나머지는 외부로 (docs/ROLES.md 4절 · 교안 p.122 "State 는 인터페이스이지 컨테이너가 아님").

    trace_id = new_trace_id()                              # "20261007-134502-a1b2c3"
    app.stream(state, config=run_config(trace_id, RECURSION_LIMIT))
    log_decision(state, "supervisor", "market", "미수집 셀 market:KIVI")
    log_retrieval(state.get("trace_id", ""), entries)

같은 trace_id 가 세 곳을 잇는다 (README State Schema 「상관」)
- 체크포인터 thread_id   — 재개 지점 (app.py --resume <trace_id>)
- LangSmith run metadata — 트레이스 검색 (metadata.trace_id) · 결정 사유는 해당 노드 run 의 metadata 로
- outputs/decisions.jsonl · outputs/retrieval_log.jsonl — {trace_id, node, ...} 한 줄씩
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

OUT = Path("outputs")
DECISIONS = "decisions.jsonl"
RETRIEVAL = "retrieval_log.jsonl"
PROJECT = "kv-cache-agent"          # LangSmith 프로젝트 이름은 .env 의 LANGSMITH_PROJECT 가 정한다 — 여기선 태그로만


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_trace_id() -> str:
    """타임스탬프 + 짧은 UUID — 사람이 읽을 수 있고 실행끼리 겹치지 않는다."""
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def run_config(trace_id: str, recursion_limit: int) -> dict[str, Any]:
    """graph.stream / invoke 의 config — 체크포인터 thread_id 와 LangSmith metadata 에 같은 trace_id."""
    return {
        "configurable": {"thread_id": trace_id},
        "metadata": {"trace_id": trace_id},
        "tags": [PROJECT, trace_id],
        "run_name": f"{PROJECT} {trace_id}",
        "recursion_limit": recursion_limit,
    }


def _append(name: str, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _langsmith_metadata(meta: dict[str, Any]) -> None:
    """지금 실행 중인 노드의 LangSmith run 에 metadata 를 단다. 트레이싱이 꺼져 있으면 아무것도 안 한다."""
    try:
        from langsmith.run_helpers import get_current_run_tree

        rt = get_current_run_tree()
        if rt is not None:
            rt.add_metadata(meta)
    except Exception as e:  # noqa: BLE001 — 관측 실패가 그래프를 멈추면 안 된다
        print(f"[observe] LangSmith metadata 실패: {type(e).__name__}: {e}", file=sys.stderr)


def log_decision(state: dict, node: str, decision: str, reason: str) -> None:
    """라우팅 · 재작업 · 종료 결정과 사유. supervisor · end_with_warning 이 부른다."""
    rec = {
        "trace_id": state.get("trace_id", ""),
        "step": state.get("step_count", 0),
        "node": node,
        "decision": decision,
        "reason": reason,
        "ts": _now(),
    }
    _append(DECISIONS, [rec])
    _langsmith_metadata({"decision": decision, "reason": reason, "step": rec["step"]})
    print(f"[{node}] step {rec['step']} → {decision}: {reason}", file=sys.stderr)


def log_retrieval(trace_id: str, entries: list[dict[str, Any]]) -> None:
    """RAG 검색 로그 (RetrievalEntry) — 재작성 전/후. State 에 쌓지 않고 파일로."""
    _append(RETRIEVAL, [{"trace_id": trace_id, "ts": _now(), **e} for e in entries])


def read_jsonl(name: str, trace_id: str | None = None) -> list[dict[str, Any]]:
    """보고서 한계점 · README Run Record 가 읽는다. trace_id 를 주면 그 실행 것만."""
    path = OUT / name
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if trace_id is None or r.get("trace_id") == trace_id]
