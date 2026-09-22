"""이해관계자 평가 에이전트 (웹) — 설계서 2장, 4.3 Rubric (찬·반 각 ≥2 — retry 시 '반대 근거 검색' 지시, 실패도 기록).  [소유: B 심준용]

출력 키: stakeholder_eval · citations · llm_calls
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, Field

from agents import _web_eval as web
from graph.dispatcher import LLM_BUDGET
from graph.state import GraphState

FAILURE = "반대 근거 확보 실패 [추론]"


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


def queries(tech: str, retry: int) -> list[str]:
    if retry:
        focus = "criticism limitations adoption barriers" if retry == 1 else "unresolved issues compatibility reproduction failures"
        return [
            f'"{tech}" KV cache {focus} accuracy overhead',
            f'"{tech}" KV cache site:github.com issues {focus}',
            f'"{tech}" KV cache independent review {focus}',
        ]
    competitor = "TurboQuant KVTC" if tech == "KIVI" else '"LLM in a flash" "LPDDR-PIM"'
    return [
        f'"{tech}" KV cache {competitor} comparison recognition criticism',
        f'"{tech}" KV cache developer adoption benefits experience',
        f'"{tech}" KV cache developer issues accuracy complexity hardware limitations',
        f'"{tech}" KV cache investor media analysis opportunity risks',
        f'"{tech}" KV cache {competitor} benefits supporting evidence',
    ]


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
            # Dispatcher가 len(negatives)를 사용하므로 중간 실패 표시는 여기에만 쓴다.
            web.add_note(result, "반대 근거 2건 미만 — 재검색 필요")


def _attempt(tech: str, summary: dict, retry: int, previous: dict) -> tuple[dict, list, int]:
    sources, notes = web.search(queries(tech, retry))
    calls, refs = 0, []
    if not sources:
        result = web.blank("이해관계자 웹 근거 없음")
    else:
        try:
            model = web.generator(StakeholderResponse)
            prompt = web.messages(
                "stakeholder", "rubrics/4.3-stakeholder", tech, summary, sources,
                mode="negative_only" if retry else "balanced", retry=retry,
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
    retry = state.get("retry", {}).get("stake", 0)
    for tech in web.TECHS:
        previous = deepcopy(state.get("stakeholder_eval", {}).get(tech, {})) if retry else {}
        if previous and len([n for n in previous["negatives"] if n != FAILURE]) >= 2:
            out[tech] = previous
            continue
        summary = state.get("tech_summary", {}).get(tech)
        if summary:
            result, refs, attempted = _attempt(tech, summary, retry, previous)
            calls += attempted
            citations.extend(refs)
        else:
            result = _merge(previous, web.blank("기술 조사 입력 없음"))
        out[tech] = result
    # 한 기술 처리 중 예산 초과가 발생해도 양쪽 모두 종료 상태를 정확하게 기록한다.
    exhausted = retry >= 2 or state.get("llm_calls", 0) + calls > LLM_BUDGET   # 상한은 dispatcher 한 곳에서 (#29: 100 → 150)
    for result in out.values():
        _finish(result, exhausted)
    return {"stakeholder_eval": out,
            "citations": web.new_citations(citations, state.get("citations", [])),
            "llm_calls": calls}
