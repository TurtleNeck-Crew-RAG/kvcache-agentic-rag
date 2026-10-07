"""State 스키마 — Agent 과제 계약 (docs/ROLES.md 4절).  [소유: D 황재원]

키 추가·이름 변경은 D 에게 이슈 — 전 워커 · 판정 노드 · Supervisor 가 의존한다. ROLES.md 4절과 같은 PR 에서만 바꾼다.

세 구역
- 페이로드 (작업 결과): 워커가 쓴다. 자기 출력 키 + citations + llm_calls 만
- 판정 (Judge 가 쓰고 Gate 가 읽는다): assess → sufficiency, evaluator → eval_result
- 제어 (Gate 만 쓴다 · safe 는 node_status / errors / last_error 만): next · rework_request · step_count · retry …

원칙
- 같은 슈퍼스텝에 둘 이상이 쓸 수 있는 키는 전부 리듀서 (리듀서 없는 키에 둘이 쓰면 InvalidUpdateError)
- *_eval · tech_summary 는 기술 단위 병합 — 재작업 워커가 한 기술만 돌려도 다른 기술 결과가 남는다
- State 에 원문 청크 · 보고서 본문을 쌓지 않는다. evidence 는 인용 문장 + 페이지, 보고서는 report_uri 로 참조만
- 결정 로그(라우팅 사유)는 State 가 아니라 graph/observe.py 가 외부에 적재한다
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

Tech = Literal["KIVI", "InfiniGen"]
TECHS: tuple[Tech, ...] = ("KIVI", "InfiniGen")
SourceTag = Literal["논문", "웹", "추론"]
Status = Literal["RUNNING", "SUCCESS", "FAILED", "INTERRUPTED"]


# ── 페이로드 타입 ────────────────────────────────────────

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
    negatives: list[str]
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
    """RAG 노드 로그 — A 가 outputs/retrieval_log.jsonl 로 외부화한다 (ROLES.md 2절 A)."""
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


# ── 판정 타입 (Judge 노드 → Gate) ────────────────────────

class CellVerdict(TypedDict):
    """셀 "{worker}:{tech}" 하나의 충분성 판정 — assess(A) 가 쓴다.

    gap — 무엇이 모자란가. 게이트가 rework_request.gap 으로 그대로 넘기고, 워커는 이 값으로 재작업 방식을 고른다
      규칙 층 (graph/sufficiency.py)
        failed       워커 실패 (node_status · safe fallback)  → 그 기술 전체 재조사
        missing      미수집
        no_evidence  등급 과반이 "근거 없음" / 개요 비어 있음
        evidence     근거([논문] · [웹]) 건수 부족
        numbers      실험 수치 없음 (tech_research)
        limitations  한계 없음 (tech_research)
        negatives    반대 근거 부족
        source_bias  출처 수 부족 · 한 출처 편중 (시장 · 이해관계자)
        counter_example  웹 근거 없음 (도메인 — #76) → 웹 반례 검색만 다시
      Judge 층 (prompts/sufficiency_judge.md)
        unsupported  주장을 근거가 받치지 않음
        off_topic    다른 관점의 근거로 채워짐
        negatives    반대 근거가 형식적
        bias         우열 판정 · 추천
      게이트 (graph/supervisor.py) — CellVerdict 가 아니라 rework_request 에만
        eval         보고서 품질 평가 fail 로 되돌린 재조사. hint_query = eval_result.feedback
    """
    rule: Literal["pass", "fail"]                                # 결정론 층
    judge: Literal["sufficient", "insufficient"] | None          # LLM Judge — 규칙 fail 이면 None
    gap: str                                                     # 위 목록 — 충분하면 ""
    hint_query: str                                              # 재작업 때 보강 질의
    reason: str


class EvalItem(TypedDict):
    passed: bool
    score: float
    reason: str


class EvalItems(TypedDict):
    groundedness: EvalItem
    neutrality: EvalItem
    bias: EvalItem
    coverage: EvalItem


class EvalResult(TypedDict):
    """보고서 품질 판정 — evaluator(C) 가 쓴다. targets 는 "report" 또는 "{worker}:{tech}"."""
    passed: bool
    items: EvalItems
    targets: list[str]
    feedback: str


# ── 제어 타입 ───────────────────────────────────────────

class ReworkRequest(TypedDict):
    """Gate → 워커. 워커는 worker 가 자기 이름일 때만 읽고 그 기술만 다시 돌린다."""
    worker: str
    tech: str
    gap: str
    hint_query: str


class ErrorRecord(TypedDict):
    node: str
    type: str
    message: str
    ts: str


# ── 리듀서 ──────────────────────────────────────────────

def merge(a: dict | None, b: dict | None) -> dict:
    """키 병합 — 덮어쓰기 방지. sufficiency · retry · node_status."""
    return {**(a or {}), **(b or {})}


def merge_by_tech(a: dict | None, b: dict | None) -> dict:
    """기술 단위 병합 — {"InfiniGen": …} 만 돌아와도 KIVI 결과는 남는다."""
    return {**(a or {}), **(b or {})}


class GraphState(TypedDict, total=False):
    # ── 페이로드 ──
    domain: dict[str, Any]                       # config/domain.yaml
    selected: dict[str, Any]                     # config/selection.yaml
    tech_summary: Annotated[dict[Tech, TechSummary], merge_by_tech]
    market_eval: Annotated[dict[Tech, Eval], merge_by_tech]
    stakeholder_eval: Annotated[dict[Tech, Eval], merge_by_tech]
    domain_eval: Annotated[dict[Tech, DomainEval], merge_by_tech]
    trl_estimate: dict[Tech, TRL]
    synthesis: Synthesis | None
    report_uri: str | None                       # 본문은 파일 (outputs/report/report.md) — State 는 참조만
    citations: Annotated[list[Ref], operator.add]

    # ── 판정 ──
    sufficiency: Annotated[dict[str, CellVerdict], merge]   # "market:InfiniGen" → CellVerdict
    eval_result: EvalResult | None

    # ── 제어 ──
    trace_id: str                                # = 체크포인터 thread_id = LangSmith metadata
    next: str                                    # 매 턴 1개
    rework_request: ReworkRequest | None
    step_count: Annotated[int, operator.add]
    max_steps: int
    retry: Annotated[dict[str, int], merge]      # "market:InfiniGen" → 재작업 횟수
    eval_attempts: int
    llm_calls: Annotated[int, operator.add]
    status: Status
    node_status: Annotated[dict[str, str], merge]           # node → "ok" | "failed"
    errors: Annotated[list[ErrorRecord], operator.add]
    last_error: ErrorRecord | None

    # ── 이행 중 (RAG 과제 키) — 소유자가 옮기면 삭제 ──
    report_md: str | None                        # B report → report_uri 로 이행 후 삭제
    neutrality: Neutrality                       # C 중립성 Judge → evaluator 통합 여부 결정 후 정리
    retrieval_log: Annotated[list[RetrievalEntry], operator.add]   # A → outputs/retrieval_log.jsonl 로 이행 후 삭제


def init_state(domain: dict[str, Any], selected: dict[str, Any], *,
               trace_id: str = "", max_steps: int = 30) -> GraphState:
    """max_steps 기본값은 graph/supervisor.py 의 MAX_STEPS 와 같게 둔다 (app.py 가 상수를 넘긴다)."""
    return GraphState(
        domain=domain,
        selected=selected,
        tech_summary={},
        market_eval={},
        stakeholder_eval={},
        domain_eval={},
        trl_estimate={},
        synthesis=None,
        report_uri=None,
        citations=[],
        sufficiency={},
        eval_result=None,
        trace_id=trace_id,
        next="",
        rework_request=None,
        step_count=0,
        max_steps=max_steps,
        retry={},
        eval_attempts=0,
        llm_calls=0,
        status="RUNNING",
        node_status={},
        errors=[],
        last_error=None,
        report_md=None,
        retrieval_log=[],
    )
