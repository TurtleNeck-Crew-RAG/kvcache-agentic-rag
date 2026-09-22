"""State 스키마 — 설계서 5.2 표를 그대로 옮김.  [소유: D 황재원]

✅ 확정 2026-09-22 (#4). 이후 키 추가·이름 변경은 D 에게 이슈 — 전 워커가 의존한다.

원칙
- fan-out 3개(market / stakeholder / domain)는 분리 키 → 동시 갱신 충돌 없음
- 누적(reducer)은 citations · retrieval_log · llm_calls · retry 만
- State 에 원문 청크를 쌓지 않는다. evidence 에 인용 문장 + 페이지만
- 모든 워커는 run(state) -> dict (갱신할 키만 반환)
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

Tech = Literal["KIVI", "InfiniGen"]
SourceTag = Literal["논문", "웹", "추론"]


class Evidence(TypedDict):
    claim: str
    tag: SourceTag
    ref: str            # 논문이면 arXiv id, 웹이면 URL
    page: int | None    # [p.N] — 논문일 때만


class Eval(TypedDict):
    """시장 · 이해관계자 평가 공통 (설계서 5.2)."""
    grade: str
    rationale: str
    positives: list[str]
    negatives: list[str]          # 이해관계자는 ≥2 강제 (Dispatcher 규칙 2')
    evidence: list[Evidence]
    confidence: float


class DomainEval(Eval, total=False):
    """도메인 평가 = Eval + 3축 트레이드오프 + 판정 (적합 / 조건부 / 부적합)."""
    verdict: Literal["적합", "조건부", "부적합"]
    axes: dict[str, str]          # recall / latency / memory → 포기한 것


class TechSummary(TypedDict):
    overview: str
    mechanism: str
    numbers: list[str]            # 수치 + [p.N]
    limitations: list[str]
    apply_conditions: list[str]
    evidence: list[Evidence]


class Ref(TypedDict):
    """citations 항목 — 설계서 6장 REFERENCE 스키마."""
    type: Literal["논문", "특허", "웹"]
    authors: str
    year: str
    title: str
    venue: str
    id_or_url: str
    accessed: str


class RetrievalEntry(TypedDict):
    """RAG 노드가 남기는 로그 — 재작성 전/후를 둘 다 기록 (설계서 5.2)."""
    node: str
    tech: Tech
    query_before: str
    query_after: str | None
    hits_before: list[str]        # chunk_id
    hits_after: list[str] | None
    relevance: Literal["yes", "no", "no_evidence"]
    rewritten: bool


class TRL(TypedDict):
    level: int
    basis: list[str]
    reference_date: str


class Synthesis(TypedDict):
    matrix: dict[str, dict[Tech, str]]   # 관점 → 기술 → 요약
    agreements: list[str]
    conflicts: list[str]


class Neutrality(TypedDict):
    result: Literal["pass", "fail"]
    violations: list[str]


def _merge_retry(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    """키 병합 — 덮어쓰기 방지 (설계서 5.2)."""
    return {**a, **b}


class GraphState(TypedDict, total=False):
    # 입력
    domain: dict[str, Any]                       # config/domain.yaml
    selected: dict[str, Any]                     # config/selection.yaml
    # 워커 출력 (분리 키)
    tech_summary: dict[Tech, TechSummary]
    market_eval: dict[Tech, Eval]
    stakeholder_eval: dict[Tech, Eval]
    domain_eval: dict[Tech, DomainEval]
    trl_estimate: dict[Tech, TRL]
    synthesis: Synthesis
    neutrality: Neutrality
    report_md: str
    # 누적
    citations: Annotated[list[Ref], operator.add]
    retrieval_log: Annotated[list[RetrievalEntry], operator.add]
    llm_calls: Annotated[int, operator.add]
    retry: Annotated[dict[str, int], _merge_retry]
    # 제어
    next: list[str]


def init_state(domain: dict[str, Any], selected: dict[str, Any]) -> GraphState:
    return GraphState(
        domain=domain,
        selected=selected,
        tech_summary={},
        market_eval={},
        stakeholder_eval={},
        domain_eval={},
        trl_estimate={},
        citations=[],
        retrieval_log=[],
        llm_calls=0,
        retry={},
        next=[],
    )
