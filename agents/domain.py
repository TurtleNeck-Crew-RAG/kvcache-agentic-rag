"""도메인 평가 에이전트 — 설계서 2장, 4.4 Rubric.  [소유: C 민영은]

1단계에서 논문의 기술 사실을 RAG로 추출하고, Tavily로 HW 반례를
수집한 뒤 2단계에서 스마트폰 배포 제약을 기준으로 적용 가능성을 판정한다.
추가 편향 검증 장치와 중립성 Judge는 #15에서 결합한다.

출력 키: domain_eval · citations · retrieval_log · llm_calls
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Literal
from urllib.parse import urlparse

from langchain_tavily import TavilySearch
from pydantic import BaseModel, ConfigDict, Field

from agents._common import TECHS, llm, load_prompt
from graph.state import Evidence, GraphState, Ref, RetrievalEntry
from rag.rag_node import ask

FACT_QUESTIONS = (
    "KV cache 메모리 사용량 절감률과 그 수치가 성립하는 설정·조건은?",
    "적용에 재학습, 파인튜닝 또는 보정 데이터가 필요한가?",
    "추가 하드웨어, 메모리 계층 또는 대역폭 전제는 무엇인가?",
    "정확도 손실 수치와 그 실험 조건은 무엇인가?",
    "전송, 프리패치 또는 추가 연산 오버헤드는 무엇인가?",
)


class EvidenceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1)
    tag: Literal["논문", "웹", "추론"]
    ref: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)


class AxesOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recall: str = Field(min_length=1)
    latency: str = Field(min_length=1)
    memory: str = Field(min_length=1)


class DomainEvaluationOutput(BaseModel):
    """기술 1개의 도메인 판정 structured output."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["적합", "조건부", "부적합"]
    rationale: str = Field(min_length=1)
    positives: list[str] = Field(min_length=1)
    negatives: list[str] = Field(min_length=1)
    axes: AxesOutput
    evidence: list[EvidenceOutput] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


def _render_prompt(
    tech: str,
    domain_spec: dict[str, Any],
    facts: list[dict[str, Any]],
    web_evidence: list[dict[str, Any]],
) -> str:
    constraints = {
        key: value for key, value in domain_spec.items() if key != "counter_examples"
    }
    return "\n\n".join(
        (
            load_prompt("domain"),
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
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for name in counter_examples:
        query = f"{name} mobile LLM memory offloading bandwidth latency hardware evaluation"
        raw = search.invoke({"query": query})
        for item in _normalise_search_results(raw):
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


def run(state: GraphState) -> dict:
    """도메인 사실 추출과 배포 제약 판정을 순서대로 실행한다."""

    domain_spec = state.get("domain", {})
    if not domain_spec:
        raise ValueError("domain worker requires state['domain']")

    evaluator = llm("generator").with_structured_output(DomainEvaluationOutput)
    evaluations: dict[str, dict[str, Any]] = {}
    retrieval_log: list[RetrievalEntry] = []
    citations: list[Ref] = []
    llm_calls = 0

    for tech in TECHS:
        facts: list[dict[str, Any]] = []

        for question in FACT_QUESTIONS:
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
            llm_calls += int(result.get("llm_calls", 0))

        web_evidence: list[dict[str, Any]] = []
        if tech == "InfiniGen":
            web_search = TavilySearch(
                max_results=3,
                search_depth="advanced",
                include_answer=False,
            )
            counter_examples = [
                str(item) for item in domain_spec.get("counter_examples", [])
            ]
            web_evidence = _search_counter_examples(web_search, counter_examples)

        prompt = _render_prompt(tech, domain_spec, facts, web_evidence)
        structured = evaluator.invoke(prompt)
        if isinstance(structured, dict):
            structured = DomainEvaluationOutput.model_validate(structured)
        evaluation = _as_domain_eval(structured)
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

    return {
        "domain_eval": evaluations,
        "citations": citations,
        "retrieval_log": retrieval_log,
        "llm_calls": llm_calls,
    }
