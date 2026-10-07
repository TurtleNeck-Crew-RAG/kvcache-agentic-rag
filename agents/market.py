"""시장 평가 에이전트 (웹) — 설계서 2장, 4.2 Rubric (Tavily 만, RAG 없음, 기술별 독립 호출).  [소유: B 심준용]

출력 키: market_eval · citations · llm_calls
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from agents import _web_eval as web
from graph.state import GraphState


class MarketResponse(BaseModel):
    adoption: web.MarketAxis
    market_connection: web.MarketAxis
    ecosystem: web.MarketAxis
    positives: list[web.Claim]
    negatives: list[web.Claim]
    confidence: float = Field(ge=0, le=1)


def queries(tech: str, hint_query: str = "") -> list[str]:
    base = [
        f'"{tech}" KV cache official adoption framework integration release',
        f'"{tech}" KV cache on-device LLM market memory demand',
        f'"{tech}" KV cache open source implementation roadmap follow-up research',
    ]
    if hint_query.strip():
        return [f'"{tech}" KV cache {hint_query.strip()}', *base]
    return base


def _targets(state: GraphState) -> tuple[tuple[str, ...], str]:
    """시장 재작업이면 요청된 기술만, 아니면 초기 실행처럼 두 기술을 처리한다."""
    request = state.get("rework_request")
    if not request or request.get("worker") != "market":
        return web.TECHS, ""

    tech = request.get("tech")
    if tech not in web.TECHS:
        raise ValueError(f"unknown market rework technology: {tech!r}")
    return (tech,), request.get("hint_query", "")


def _evaluate(response: MarketResponse, sources: dict) -> tuple[dict, list]:
    grounding = web.Grounding(sources)
    grades, reasons = [], []
    for name, axis in (("채택", response.adoption), ("시장 연결", response.market_connection),
                       ("생태계", response.ecosystem)):
        reason = grounding.claim(axis.reason)
        grade = axis.grade if reason and axis.reason.source_url else "근거 없음"
        grades.append(f"{name}: {grade}")
        reasons.append(f"{name}: {reason or '유효한 근거 없음 [추론]'}")
    positives = grounding.claims(response.positives)
    negatives = grounding.claims(response.negatives)
    result = {
        "grade": " / ".join(grades), "rationale": "\n".join(reasons),
        "positives": positives, "negatives": negatives, "evidence": grounding.evidence,
        "confidence": min(response.confidence, 0.3) if grounding.rejected else response.confidence,
    }
    if grounding.rejected:
        web.add_note(result, "검색 결과로 확인되지 않은 근거 제외")
    return result, grounding.citations()


def run(state: GraphState) -> dict:
    out, citations, calls = {}, [], 0
    targets, hint_query = _targets(state)
    for tech in targets:
        summary = state.get("tech_summary", {}).get(tech)
        if not summary:
            out[tech] = web.blank("기술 조사 입력 없음 — fixtures는 테스트에서 명시적으로 주입")
            continue
        sources, notes = web.search(queries(tech, hint_query), node="market", trace_id=state.get("trace_id", ""))
        if not sources:
            out[tech] = web.blank("시장 웹 근거 없음")
        else:
            try:
                model = web.generator(MarketResponse)
                prompt = web.messages("market", "rubrics/4.2-market", tech, summary, sources)
                calls += 1
                response = MarketResponse.model_validate(model.invoke(prompt))
                out[tech], refs = _evaluate(response, sources)
                citations.extend(refs)
            except Exception as exc:
                out[tech] = web.blank(f"시장 평가 생성 실패 ({type(exc).__name__})")
        for note in notes:
            web.add_note(out[tech], note)
    return {"market_eval": out, "citations": web.new_citations(citations, state.get("citations", [])),
            "llm_calls": calls}
