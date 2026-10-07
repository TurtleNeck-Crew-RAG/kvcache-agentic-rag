"""도메인 평가 에이전트 — 설계서 2장, 4.4 Rubric.  [소유: C 민영은]

1단계에서 논문의 기술 사실을 RAG로 추출하고, Tavily로 HW 반례를
수집한 뒤 2단계에서 스마트폰 배포 제약을 기준으로 적용 가능성을 판정한다.

재작업(rework_request.worker == "domain")이면 요청된 기술 하나만 다시 돈다 (#81).
- 이전 결과가 없거나 믿을 수 없는 gap(missing · failed · no_evidence) → 그 기술만 처음부터
- 그 밖에는 기존 평가 + hint_query 보강 검색(웹 반례 · 출처 gap 은 웹, 나머지는 논문 RAG)으로 DomainEval 재생성
- 반환 domain_eval 에는 요청 기술만 — merge_by_tech 리듀서가 다른 기술 결과를 지킨다

출력 키: domain_eval · citations · llm_calls — 검색 로그는 outputs/retrieval_log.jsonl, 웹 검색 횟수는 web_calls.jsonl (graph/observe)
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

from langchain_tavily import TavilySearch
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from agents._common import TECHS, llm, load_prompt
from graph.observe import log_retrieval, log_web
from graph.state import Evidence, GraphState, Ref, RetrievalEntry
from rag.rag_node import ask

FACT_QUESTIONS = (
    "KV cache 메모리 사용량 절감률과 그 수치가 성립하는 설정·조건은?",
    "적용에 재학습, 파인튜닝 또는 보정 데이터가 필요한가?",
    "추가 하드웨어, 메모리 계층 또는 대역폭 전제는 무엇인가?",
    "정확도 손실 수치와 그 실험 조건은 무엇인가?",
    "전송, 프리패치 또는 추가 연산 오버헤드는 무엇인가?",
)
FULL_RERUN_GAPS = {"missing", "failed", "no_evidence"}            # 이전 결과를 근거로 쓸 수 없다
WEB_GAPS = {"counter_example", "source_bias", "bias", "negatives"}  # 웹 반례 · 출처 다양성 · 반대 근거
PAPER_GAPS_SKIP = {"counter_example", "source_bias", "bias"}         # 논문 RAG 로는 못 메우는 gap
MAX_HINT_CHARS = 200                                                 # evaluator feedback 이 hint 로 오면 길다
SOURCE_TAG = re.compile(r"\[(?:논문|웹|추론|p\.\d)[^\]]*\]")   # [논문 p.2, p.9] · [논문 2406.19707 p.9] · [p.3] · [웹 URL] · [추론] — 실출력은 쪽을 여러 개 묶는다(5회차)


BARE_URL_TAG = re.compile(r"\[(https?://[^\]\s]+)\]")


def _require_source_tag(value: str) -> str:
    value = BARE_URL_TAG.sub(r"[웹 \1]", value)   # [URL] → [웹 URL] — 다른 워커 · README 출처 태그 형식으로 통일 (#130)
    if "근거 없음" not in value and not SOURCE_TAG.search(value):   # "…(논문에 근거 없음)." 도 근거 없음 기록으로 인정
        # 태그 없는 판단 문장 = 근거 없는 추론으로 기록한다 (장치 6). 예외로 워커 전체를 버리면 태그가 있는 나머지 문장까지
        # 사라진다 — 6회차: TRL basis 2문장 때문에 종합 전체가 실패. [추론] 은 한계점 4 의 비율에 그대로 잡힌다.
        return value.rstrip() + " [추론]"
    return value


TaggedText = Annotated[str, AfterValidator(_require_source_tag)]


class EvidenceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1)
    tag: Literal["논문", "웹", "추론"]
    ref: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)


class AxesOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recall: TaggedText
    latency: TaggedText
    memory: TaggedText


class DomainEvaluationOutput(BaseModel):
    """기술 1개의 도메인 판정 structured output."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["적합", "조건부", "부적합"]
    rationale: TaggedText
    positives: list[TaggedText] = Field(min_length=1)
    negatives: list[TaggedText] = Field(min_length=2)
    axes: AxesOutput
    evidence: list[EvidenceOutput] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


def _render_prompt(
    tech: str,
    domain_spec: dict[str, Any],
    facts: list[dict[str, Any]],
    web_evidence: list[dict[str, Any]],
    rework: dict[str, Any] | None = None,
) -> str:
    constraints = {
        key: value for key, value in domain_spec.items() if key != "counter_examples"
    }
    rework_parts = ()
    if rework:
        rework_parts = (
            "## 재작업 요청\n"
            + json.dumps(
                {"gap": rework["gap"], "hint_query": rework["hint"], "previous": rework["previous"]},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
        )
    return "\n\n".join(
        (
            load_prompt("domain"),
            *rework_parts,
            "## Rubric\n" + load_prompt("rubrics/4.4-domain"),
            f"## 평가 기술\n{tech}",
            "## 도메인 제약\n"
            + json.dumps(constraints, ensure_ascii=False, indent=2, default=str),
            "## 논문 사실 추출 결과\n"
            + json.dumps(facts, ensure_ascii=False, indent=2, default=str),
            "## HW 반례 웹 검색 결과\n"
            + json.dumps(web_evidence, ensure_ascii=False, indent=2, default=str),
        )
    )


def _normalise_search_results(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        raw = raw.get("results", [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict) and item.get("url")]


def _search_counter_examples(
    search: TavilySearch,
    counter_examples: list[str],
    extra_queries: tuple[str, ...] = (),
    trace_id: str = "",
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    jobs = [
        (name, f"{name} mobile LLM memory offloading bandwidth latency hardware evaluation")
        for name in counter_examples
    ] + [("rework", query) for query in extra_queries]
    for name, query in jobs:
        try:
            raw = search.invoke({"query": query})
        except Exception:
            log_web(trace_id, "domain", query, 0)                 # 실패해도 크레딧은 쓰였다 (#102)
            raise
        items = _normalise_search_results(raw)
        log_web(trace_id, "domain", query, len(items))
        for item in items:
            collected.append(
                {
                    "counter_example": name,
                    "title": str(item.get("title", "")),
                    "url": str(item["url"]),
                    "content": str(item.get("content", "")),
                    "score": item.get("score"),
                }
            )
    return collected


def _selection_for_tech(state: GraphState, tech: str) -> dict[str, Any]:
    for side in ("sw", "hw"):
        selected = state.get("selected", {}).get(side, {})
        if selected.get("name") == tech:
            return selected
    return {}


def _paper_ref(state: GraphState, tech: str, evidence: list[Evidence]) -> Ref | None:
    paper_evidence = next((item for item in evidence if item.get("tag") == "논문"), None)
    if not paper_evidence:
        return None

    selected = _selection_for_tech(state, tech)
    arxiv = str(selected.get("arxiv", paper_evidence.get("ref", "")))
    venue = str(selected.get("venue", ""))
    year_match = re.search(r"\b(19|20)\d{2}\b", venue)
    return {
        "type": "논문",
        "authors": str(selected.get("authors", "")),
        "year": year_match.group(0) if year_match else "",
        "title": str(selected.get("paper", tech)),
        "venue": venue,
        "id_or_url": arxiv,
        "accessed": date.today().isoformat(),
    }


def _web_refs(
    web_evidence: list[dict[str, Any]],
    referenced_urls: set[str],
) -> list[Ref]:
    accessed = date.today().isoformat()
    refs: dict[str, Ref] = {}
    for item in web_evidence:
        url = str(item["url"])
        if url not in referenced_urls:
            continue
        host = urlparse(url).netloc.removeprefix("www.")
        refs.setdefault(
            url,
            {
                "type": "웹",
                "authors": host,
                "year": "",
                "title": str(item.get("title", url)),
                "venue": host,
                "id_or_url": url,
                "accessed": accessed,
            },
        )
    return list(refs.values())


def _as_domain_eval(value: DomainEvaluationOutput) -> dict[str, Any]:
    data = value.model_dump()
    data["grade"] = data["verdict"]
    return data


def _new_search() -> TavilySearch:
    return TavilySearch(max_results=3, search_depth="advanced", include_answer=False)


def _rework_request(state: GraphState) -> dict[str, Any] | None:
    request = state.get("rework_request") or {}
    if request.get("worker") != "domain":
        return None
    if request.get("tech") not in TECHS:
        raise ValueError(f"unknown domain rework technology: {request.get('tech')!r}")
    return request


def _clip_hint(hint: str) -> str:
    """hint_query 첫 줄만, 검색 질의 길이로 자른다 (evaluator feedback 은 여러 줄)."""
    line = next((part.strip() for part in str(hint or "").splitlines() if part.strip()), "")
    return line[:MAX_HINT_CHARS]


def _ask_fact(tech: str, question: str, facts: list, retrieval_log: list) -> int:
    result = ask(tech, question, node="domain")
    facts.append(
        {
            "question": question,
            "answer": result.get("answer", "논문에 근거 없음"),
            "evidence": result.get("evidence", []),
        }
    )
    entry = result.get("retrieval_entry")
    if entry:
        retrieval_log.append(entry)
    return int(result.get("llm_calls", 0))


def _merge_evidence(previous: list[dict], new: list[dict]) -> list[dict]:
    """재작업 결과에 기존 근거를 남긴다 — 재생성이 근거를 덜 고르면 충분성이 오히려 떨어진다."""
    merged: dict[tuple, dict] = {}
    for item in [*new, *previous]:
        merged.setdefault((item.get("tag"), str(item.get("ref")), item.get("claim")), item)
    return list(merged.values())


def run(state: GraphState) -> dict:
    """도메인 사실 추출과 배포 제약 판정을 순서대로 실행한다. 재작업이면 요청 기술만."""

    domain_spec = state.get("domain", {})
    if not domain_spec:
        raise ValueError("domain worker requires state['domain']")

    request = _rework_request(state)
    techs = (request["tech"],) if request else TECHS
    evaluator = llm("generator").with_structured_output(DomainEvaluationOutput)
    evaluations: dict[str, dict[str, Any]] = {}
    retrieval_log: list[RetrievalEntry] = []
    citations: list[Ref] = []
    llm_calls = 0
    counter_examples = [str(item) for item in domain_spec.get("counter_examples", [])]

    for tech in techs:
        previous = (state.get("domain_eval") or {}).get(tech) if request else None
        gap = str(request.get("gap", "")) if request else ""
        hint = _clip_hint(request.get("hint_query", "")) if request else ""
        reinforce = previous is not None and gap not in FULL_RERUN_GAPS and bool(hint)
        facts: list[dict[str, Any]] = []
        web_evidence: list[dict[str, Any]] = []

        if reinforce:
            # 기존 평가 + 보강 근거 — 사실 질의 5개를 다시 돌리지 않는다
            if gap not in PAPER_GAPS_SKIP:
                llm_calls += _ask_fact(tech, hint, facts, retrieval_log)
            if gap in WEB_GAPS:
                web_evidence = _search_counter_examples(
                    _new_search(),
                    counter_examples if tech == "InfiniGen" else [],
                    (f"{tech} {hint}",),
                    trace_id=state.get("trace_id", ""),
                )
        else:
            for question in FACT_QUESTIONS:
                llm_calls += _ask_fact(tech, question, facts, retrieval_log)
            extra = (f"{tech} {hint}",) if request and hint else ()
            if tech == "InfiniGen" or extra:
                web_evidence = _search_counter_examples(
                    _new_search(), counter_examples if tech == "InfiniGen" else [], extra,
                    trace_id=state.get("trace_id", ""),
                )

        rework = {"gap": gap, "hint": hint, "previous": previous} if reinforce else None
        prompt = _render_prompt(tech, domain_spec, facts, web_evidence, rework)
        structured = evaluator.invoke(prompt)
        if isinstance(structured, dict):
            structured = DomainEvaluationOutput.model_validate(structured)
        evaluation = _as_domain_eval(structured)
        if reinforce:
            evaluation["evidence"] = _merge_evidence(previous.get("evidence") or [], evaluation["evidence"])
        evaluations[tech] = evaluation
        llm_calls += 1

        used_evidence = evaluation["evidence"]
        citation = _paper_ref(state, tech, used_evidence)
        if citation:
            citations.append(citation)
        referenced_urls = {
            str(item["ref"]) for item in used_evidence if item["tag"] == "웹"
        }
        citations.extend(_web_refs(web_evidence, referenced_urls))

    log_retrieval(state.get("trace_id", ""), retrieval_log)
    return {
        "domain_eval": evaluations,
        "citations": citations,
        "llm_calls": llm_calls,
    }
