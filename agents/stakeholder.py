"""이해관계자 평가 에이전트 (웹) — 설계서 2장, 4.3 Rubric (찬·반 각 ≥2 — 재작업 시 부족 근거 보강).  [소유: B 심준용]

출력 키: stakeholder_eval · citations · llm_calls
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, Field

from agents import _web_eval as web
from graph.state import GraphState

FAILURE = "반대 근거 확보 실패 [추론]"
MAX_REWORKS = 2


class StakeholderGroup(BaseModel):
    stance: Literal["우호", "중립", "비판", "근거 없음"]
    reason: web.Claim


class StakeholderResponse(BaseModel):
    competitors: StakeholderGroup
    developers: StakeholderGroup
    investment_media: StakeholderGroup
    positives: list[web.Claim]
    negatives: list[web.Claim]
    confidence: float = Field(ge=0, le=1)


def queries(tech: str, *, negative_only: bool = False, hint_query: str = "") -> list[str]:
    if negative_only:
        base = [
            f'"{tech}" KV cache criticism limitations adoption barriers accuracy overhead',
            f'"{tech}" KV cache site:github.com issues unresolved compatibility reproduction failures',
            f'"{tech}" KV cache independent review criticism limitations',
        ]
    else:
        competitor = "TurboQuant KVTC" if tech == "KIVI" else '"LLM in a flash" "LPDDR-PIM"'
        base = [
            f'"{tech}" KV cache {competitor} comparison recognition criticism',
            f'"{tech}" KV cache developer adoption benefits experience',
            f'"{tech}" KV cache developer issues accuracy complexity hardware limitations',
            f'"{tech}" KV cache investor media analysis opportunity risks',
            f'"{tech}" KV cache {competitor} benefits supporting evidence',
        ]
    if hint_query.strip():
        return [f'"{tech}" KV cache {hint_query.strip()}', *base]
    return base


def _targets(state: GraphState) -> tuple[tuple[str, ...], dict | None]:
    """이해관계자 재작업이면 요청된 기술만, 아니면 두 기술을 처음 평가한다."""
    request = state.get("rework_request")
    if not request or request.get("worker") != "stakeholder":
        return web.TECHS, None

    tech = request.get("tech")
    if tech not in web.TECHS:
        raise ValueError(f"unknown stakeholder rework technology: {tech!r}")
    return (tech,), request


def _evaluate(response: StakeholderResponse, sources: dict) -> tuple[dict, list]:
    grounding = web.Grounding(sources)
    grades, reasons = [], []
    for name, group in (("경쟁 기술 진영", response.competitors),
                        ("도입 기업·개발자", response.developers),
                        ("투자·미디어", response.investment_media)):
        reason = grounding.claim(group.reason)
        stance = group.stance if reason and group.reason.source_url else "근거 없음"
        grades.append(f"{name}: {stance}")
        reasons.append(f"{name}: {reason or '유효한 근거 없음 [추론]'}")
    # 추론/실패 메시지를 찬반 근거 최소 건수로 인정하지 않는다.
    positives = grounding.claims(response.positives, require_web=True)
    # 발췌문도 남겨 다음 실행에서 같은 근거의 문구만 바꾼 중복을 식별한다.
    negative_claims = [claim.model_copy(update={"text": f"{claim.text} (원문: {grounding.quote(claim)})"})
                       for claim in response.negatives]
    negatives = grounding.claims(negative_claims, require_web=True)
    result = {
        "grade": " / ".join(grades), "rationale": "\n".join(reasons),
        "positives": positives, "negatives": negatives, "evidence": grounding.evidence,
        "confidence": min(response.confidence, 0.3) if grounding.rejected else response.confidence,
    }
    if grounding.rejected:
        web.add_note(result, "출처 없는 찬반 주장 또는 검색 결과로 확인되지 않은 근거 제외")
    return result, grounding.citations()


def _merge(previous: dict, fresh: dict) -> dict:
    """반대 근거 재검색이 기존의 찬성 근거·출처를 지우지 않게 병합한다."""
    result = deepcopy(fresh)
    if not previous:
        return result
    for key in ("positives", "negatives", "evidence"):
        merged, seen = [], set()
        for item in previous[key] + fresh[key]:
            if item == FAILURE or item in merged:
                continue
            if key == "negatives":
                match = re.search(r"\(원문: (.*)\) \[웹 (https?://[^\s\]]+)\]$", item, re.DOTALL)
                identity = (" ".join(match[1].split()).casefold(), match[2]) if match else item
                if identity in seen:
                    continue
                seen.add(identity)
            merged.append(item)
        result[key] = merged
    result["rationale"] = "이전 평가:\n" + previous["rationale"] + "\n재검색 결과:\n" + fresh["rationale"]
    if fresh["grade"] == "평가 불가":
        result["grade"] = previous["grade"]
    elif previous["grade"] != "평가 불가":
        # 반대 근거 전용 검색에서 빠진 그룹의 기존 판정은 그대로 보존한다.
        prior = dict(part.split(": ", 1) for part in previous["grade"].split(" / "))
        groups = []
        for part in fresh["grade"].split(" / "):
            name, stance = part.split(": ", 1)
            if stance == "근거 없음":
                stance = prior.get(name, stance)
            groups.append(f"{name}: {stance}")
        result["grade"] = " / ".join(groups)
    # 새 평가의 confidence를 근거 없이 상향하지 않는다.
    result["confidence"] = min(previous["confidence"], fresh["confidence"])
    return result


def _finish(result: dict, exhausted: bool) -> None:
    result["negatives"] = [item for item in result["negatives"] if item != FAILURE]
    if len(result["positives"]) < 2:
        web.add_note(result, "찬성 근거 2건 미만")
        result["confidence"] = min(result["confidence"], 0.3)
    if len(result["negatives"]) < 2:
        result["confidence"] = min(result["confidence"], 0.3)
        if exhausted:
            result["negatives"].append(FAILURE)
            web.add_note(result, "반대 근거 확보 실패")
        else:
            # assess가 실제 근거 수를 판정하므로 중간 실패 표시는 rationale에만 남긴다.
            web.add_note(result, "반대 근거 2건 미만 — 재검색 필요")


def _attempt(tech: str, summary: dict, *, negative_only: bool, hint_query: str,
             attempt: int, previous: dict) -> tuple[dict, list, int]:
    sources, notes = web.search(queries(tech, negative_only=negative_only, hint_query=hint_query))
    calls, refs = 0, []
    if not sources:
        result = web.blank("이해관계자 웹 근거 없음")
    else:
        try:
            model = web.generator(StakeholderResponse)
            prompt = web.messages(
                "stakeholder", "rubrics/4.3-stakeholder", tech, summary, sources,
                mode="negative_only" if negative_only else "balanced", retry=attempt,
                previous_eval=previous,
            )
            calls += 1
            response = StakeholderResponse.model_validate(model.invoke(prompt))
            result, refs = _evaluate(response, sources)
        except Exception as exc:
            result = web.blank(f"이해관계자 평가 생성 실패 ({type(exc).__name__})")
    for note in notes:
        web.add_note(result, note)
    return _merge(previous, result), refs, calls


def run(state: GraphState) -> dict:
    out, citations, calls = {}, [], 0
    targets, request = _targets(state)
    for tech in targets:
        is_rework = request is not None
        previous = deepcopy(state.get("stakeholder_eval", {}).get(tech, {})) if is_rework else {}
        attempt = state.get("retry", {}).get(f"stakeholder:{tech}", 0) if is_rework else 0
        gap = request.get("gap", "") if request else ""
        hint_query = request.get("hint_query", "") if request else ""
        summary = state.get("tech_summary", {}).get(tech)
        if summary:
            result, refs, attempted = _attempt(
                tech, summary,
                negative_only=is_rework and gap == "negatives",
                hint_query=hint_query,
                attempt=attempt,
                previous=previous,
            )
            calls += attempted
            citations.extend(refs)
        else:
            result = _merge(previous, web.blank("기술 조사 입력 없음"))
        _finish(result, exhausted=is_rework and attempt >= MAX_REWORKS)
        out[tech] = result
    return {"stakeholder_eval": out,
            "citations": web.new_citations(citations, state.get("citations", [])),
            "llm_calls": calls}
