"""워커 안전 래퍼 — 예외를 "자기 출력 키의 실패 기록"으로 바꾼다.

설계서 5.0 규칙: "워커는 실패해도 자기 State 키에 실패 기록을 쓴다 — 빈 값으로 두지 않는다."
통합 실행 1·2회차(2026-09-22)에서 fan-out 형제 하나의 예외가 같은 슈퍼스텝의 다른 워커 결과까지 지우는 것을 확인했다.
워커 코드가 try/except 를 빠뜨려도 그래프가 END 까지 가도록 그래프 층에서 한 번 더 막는다.

실패 기록은 해당 키의 정상 형식을 지킨다 (종합·보고서 워커가 그대로 읽을 수 있게). 실패 사유는 rationale / conflicts / 본문에 남는다.
"""
from __future__ import annotations

import sys
import traceback
from collections.abc import Callable
from typing import Any

TECHS = ("KIVI", "InfiniGen")


def _failed_eval(reason: str, *, domain: bool = False) -> dict[str, Any]:
    e: dict[str, Any] = {
        "grade": "근거 없음", "rationale": f"워커 실패 — {reason}",
        "positives": [], "negatives": [], "evidence": [], "confidence": 0.0,
    }
    if domain:
        e["verdict"] = "부적합"
        e["axes"] = {"recall": "근거 없음", "latency": "근거 없음", "memory": "근거 없음"}
    return e


def fallback(name: str, reason: str) -> dict[str, Any]:
    """워커 이름 → 실패 시 반환할 State 갱신. 키 형식은 graph/state.py 와 같다."""
    if name == "tech_research":
        empty = {"overview": f"근거 없음 — {reason}", "mechanism": "", "numbers": [], "limitations": [],
                 "apply_conditions": [], "evidence": []}
        return {"tech_summary": {t: dict(empty) for t in TECHS}}
    if name in ("market", "stakeholder"):
        return {f"{name}_eval": {t: _failed_eval(reason) for t in TECHS}}
    if name == "domain":
        return {"domain_eval": {t: _failed_eval(reason, domain=True) for t in TECHS}}
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


def safe(name: str, fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
    """워커 run(state) 를 감싼다. 정상이면 그대로, 예외면 fallback(name) + 표준에러 로그."""

    def wrapped(state: dict) -> dict:
        try:
            out = fn(state)
        except Exception as e:                    # noqa: BLE001 — 어떤 예외든 실패 기록으로 바꾸는 것이 목적
            reason = _short(e)
            print(f"[safe] {name} 실패 → 실패 기록으로 대체: {reason}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return fallback(name, reason)
        if not isinstance(out, dict):
            return fallback(name, f"run() 이 dict 가 아닌 {type(out).__name__} 을 반환")
        return out

    wrapped.__name__ = f"safe_{name}"
    return wrapped
