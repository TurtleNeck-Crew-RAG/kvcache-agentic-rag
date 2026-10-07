"""노드 안전 래퍼 — 예외를 "자기 출력 키의 실패 기록 + 제어 메타의 실패 표시"로 바꾼다.  [소유: D 황재원]

설계서 5.0 규칙: "워커는 실패해도 자기 State 키에 실패 기록을 쓴다 — 빈 값으로 두지 않는다."
통합 실행 1·2회차(2026-09-22)에서 fan-out 형제 하나의 예외가 같은 슈퍼스텝의 다른 워커 결과까지 지우는 것을 확인했다.
워커 코드가 try/except 를 빠뜨려도 그래프가 END 까지 가도록 그래프 층에서 한 번 더 막는다.

실패 기록은 해당 키의 정상 형식을 지킨다 (종합·보고서 워커가 그대로 읽을 수 있게). 실패 사유는 rationale / conflicts / 본문에 남는다.

Agent 과제 (#54) — fallback 이 정상 결과처럼 보이지 않게 제어 메타에도 남긴다 (교안 함정: except 가 오류를 삼키면 fallback 이 정상처럼 보인다)
- 실패: node_status[name] = "failed" · last_error · errors 에 ErrorRecord 추가 → assess 규칙이 "무조건 부족", supervisor 는 END 대신 end_with_warning
- 성공: node_status[name] = "ok" (재작업이 성공하면 failed 가 풀린다)
- rework_request 가 자기 것이면 fallback 은 그 기술만 쓴다 — merge_by_tech 라 다른 기술의 정상 결과를 덮지 않는다
- 판정 노드(assess · evaluator)도 감싼다. fallback 페이로드는 없고 node_status · errors 만
- 토큰 (#102): 노드 안의 LLM 호출 토큰을 get_usage_metadata_callback 으로 재서 {"tokens": n} 을 붙인다 (실패해도 쓴 만큼).
  워커 코드는 고치지 않는다. 노드 안에서 스레드를 띄우면 콜백이 전파되지 않으니 워커는 순차 호출을 유지한다
"""
from __future__ import annotations

import sys
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from langchain_core.callbacks import get_usage_metadata_callback

from graph.state import TECHS, ErrorRecord


def _failed_eval(reason: str, *, domain: bool = False) -> dict[str, Any]:
    e: dict[str, Any] = {
        "grade": "근거 없음", "rationale": f"워커 실패 — {reason}",
        "positives": [], "negatives": [], "evidence": [], "confidence": 0.0,
    }
    if domain:
        e["verdict"] = "부적합"
        e["axes"] = {"recall": "근거 없음", "latency": "근거 없음", "memory": "근거 없음"}
    return e


def fallback(name: str, reason: str, techs: tuple[str, ...] = TECHS) -> dict[str, Any]:
    """노드 이름 → 실패 시 반환할 페이로드. 키 형식은 graph/state.py 와 같다. 판정 노드는 {}."""
    if name == "tech_research":
        empty = {"overview": f"근거 없음 — {reason}", "mechanism": "", "numbers": [], "limitations": [],
                 "apply_conditions": [], "evidence": []}
        return {"tech_summary": {t: dict(empty) for t in techs}}
    if name in ("market", "stakeholder"):
        return {f"{name}_eval": {t: _failed_eval(reason) for t in techs}}
    if name == "domain":
        return {"domain_eval": {t: _failed_eval(reason, domain=True) for t in techs}}
    if name == "synthesis":
        return {
            "synthesis": {"matrix": {}, "agreements": [], "conflicts": [f"종합 워커 실패 — {reason} [추론]"]},
            "trl_estimate": {t: {"level": 0, "basis": [f"근거 없음 — {reason}"], "reference_date": ""} for t in TECHS},
            "neutrality": {"result": "pass", "violations": []},      # 재시도 루프에 걸리지 않게
        }
    if name == "report":
        return {"report_md": f"# 보고서 생성 실패\n\n{reason}\n"}
    return {}


def _short(e: BaseException, limit: int = 160) -> str:
    """보고서에 들어갈 실패 사유 — 예외 첫 줄만, 길면 자른다 (pydantic ValidationError 는 수십 줄이라 그대로 넣으면 4장이 깨진다. 실측 5회차)."""
    first = str(e).strip().splitlines()[0] if str(e).strip() else ""
    text = f"{type(e).__name__}: {first}"
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _failed(name: str, state: dict, err_type: str, reason: str) -> dict[str, Any]:
    req = state.get("rework_request") or {}
    techs = (req["tech"],) if req.get("worker") == name and req.get("tech") in TECHS else TECHS
    rec = ErrorRecord(node=name, type=err_type, message=reason, ts=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return {**fallback(name, reason, techs), "node_status": {name: "failed"}, "last_error": rec, "errors": [rec]}


def safe(name: str, fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
    """노드 run(state) 를 감싼다. 정상이면 + node_status ok, 예외면 fallback + node_status failed · last_error · errors."""

    def wrapped(state: dict) -> dict:
        with get_usage_metadata_callback() as usage:
            try:
                out = fn(state)
            except Exception as e:                # noqa: BLE001 — 어떤 예외든 실패 기록으로 바꾸는 것이 목적
                reason = _short(e)
                print(f"[safe] {name} 실패 → 실패 기록으로 대체: {reason}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                out = _failed(name, state, type(e).__name__, reason)
            else:
                if isinstance(out, dict):
                    out = {**out, "node_status": {name: "ok"}}
                else:
                    out = _failed(name, state, "TypeError", f"run() 이 dict 가 아닌 {type(out).__name__} 을 반환")
        tokens = sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values())
        return {**out, "tokens": tokens} if tokens else out

    wrapped.__name__ = f"safe_{name}"
    return wrapped
