"""assess — 근거 충분성 판정 노드 (Judge).  [소유: A 박유진]

워커가 끝날 때마다 8셀(관점 4 × 기술 2, 키 "{worker}:{tech}")의 근거 충분성을 판정해 sufficiency 에 기록만 한다.
next 는 쓰지 않는다 — 다음 경로는 graph/supervisor.py 의 게이트(순수 함수)가 sufficiency 를 읽고 정한다.

두 층 — 교안 부록 B "결정론 층이 먼저 거르고, 통과된 것만 Judge 로"
1. check_rules()  결정론. 워커 실패 · 근거 없음 · 근거 건수 · 반대 근거 · 출처 편중. 실패하면 Judge 를 부르지 않는다
2. LLM Judge     규칙을 통과한 셀만. 주장 ↔ 근거 대응 · 관점 적합성 · 반대 근거의 실질 · 우열 판정 (prompts/sufficiency_judge.md)

재판정 범위 — 방금 실행된 워커(state["next"])의 셀만 다시 본다. 재작업이면 그 기술 하나만.
안 바뀐 셀은 이전 판정을 그대로 둔다 (LLM 호출 절약 · 같은 입력에 판정이 흔들리지 않게).

    out = assess(state)   # {"sufficiency": {"market:InfiniGen": CellVerdict, ...}, "llm_calls": n}
"""
from __future__ import annotations

import re
import sys
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from agents._common import TECHS, llm, render_prompt

# 셀을 가진 워커 → 페이로드 키 · 관점 이름 (graph/supervisor.py PAYLOAD 와 같은 집합)
PAYLOAD = {"tech_research": "tech_summary", "market": "market_eval",
           "stakeholder": "stakeholder_eval", "domain": "domain_eval"}
PERSPECTIVE = {"tech_research": "기술 성숙도", "market": "시장성",
               "stakeholder": "이해관계자", "domain": "도메인 적용 (스마트폰 온디바이스)"}

# ── 결정론 층 기준 — README State Schema · 확증편향 방지에 그대로 적는다 ──
MIN_EVIDENCE = 3                                                  # 셀당 근거 문장 — 출처가 있는 [논문] · [웹] 만 센다
GROUNDED_TAGS = ("논문", "웹")                                      # [추론] · Faithfulness 미통과 메모는 근거가 아니다
SOURCE_TAG_RE = re.compile(r"\[(?:웹|논문|p\.\d|https?://)[^\]]*\]")  # 출처 태그 — [웹 URL] · [논문 p.3] · [p.3] · [URL](도메인 워커)
MIN_NEGATIVES = 2                                                 # 반대 근거 — 전 관점 공통 (RAG 확증편향 방지 장치 7). 질은 Judge 기준 3
# 출처 기준은 관점의 자료 구조에 따라 둘로 나뉜다
WEB_PERSPECTIVES = ("market", "stakeholder")                      # 웹 검색만 쓰는 관점
MIN_SOURCES = 2                                                   #   서로 다른 출처 2곳 이상
MAX_SOURCE_SHARE = 0.5                                            #   한 출처가 근거의 과반이면 편중
MIN_WEB_COUNTER = 1                                               # 도메인 = 논문 사실 추출 + 웹 반례 (RAG 설계서 4.4) → 웹 근거 1건 이상.
                                                                  # 반례인지(지지 근거가 아닌지)는 규칙이 못 본다 — Judge 기준 3
NO_EVIDENCE = "근거 없음"
MAX_JUDGE_EVIDENCE = 12                                           # Judge 에 넘길 근거 문장 상한 (컨텍스트 절약)

# 재작업 때 워커가 받을 보강 질의 — 규칙 실패는 결정론으로 만든다 (Judge 실패는 Judge 가 쓴다)
_KW = {"tech_research": "", "market": "시장 채택 도입 사례 생태계",
       "stakeholder": "업계 개발자 반응 비판", "domain": "스마트폰 온디바이스 모바일 적용"}
_TECH_Q = {
    "evidence": "{tech}의 핵심 메커니즘과 실험 결과는 무엇인가?",
    "numbers": "{tech}의 실험에서 보고된 메모리 · 지연 · 정확도 수치는?",
    "limitations": "{tech}의 한계나 성능이 떨어지는 조건은 무엇인가?",
}


class SufficiencyJudgment(BaseModel):
    """LLM Judge 출력 — 교안 p.162: Output Schema 는 Pydantic."""
    sufficient: bool
    gap: str = Field(description="none | unsupported | off_topic | negatives | bias")
    hint_query: str = Field(description="부족할 때 이 기술 하나에 대한 보강 검색 질의. 충분하면 빈 문자열")
    reason: str


# ── 결정론 층 ────────────────────────────────────────────

def source_key(ref: str) -> str:
    """근거 ref → 출처 단위. 웹은 도메인, 논문은 arXiv id. arxiv.org URL 은 논문 id 로 합친다."""
    ref = (ref or "").strip()
    if not ref:
        return ""
    if ref.startswith("http"):
        u = urlparse(ref)
        m = re.search(r"(\d{4}\.\d{4,5})", u.path)
        if "arxiv.org" in u.netloc and m:
            return f"arXiv:{m.group(1)}"
        return u.netloc.removeprefix("www.")
    m = re.search(r"(\d{4}\.\d{4,5})", ref)
    return f"arXiv:{m.group(1)}" if m else ref


def _source_bias(evidence: list[dict]) -> tuple[int, str, float]:
    """(서로 다른 출처 수, 최다 출처, 그 비중). 출처 없는 근거는 세지 않는다."""
    keys = [k for k in (source_key(e.get("ref", "")) for e in evidence) if k]
    if not keys:
        return 0, "", 0.0
    top = max(set(keys), key=keys.count)
    return len(set(keys)), top, keys.count(top) / len(keys)


def _mostly_no_evidence(grade: str) -> bool:
    """'채택: 근거 없음 / 시장 연결: 중 / 생태계: 근거 없음' — 하위 항목의 과반이 근거 없음이면 True."""
    parts = [p for p in re.split(r"\s*/\s*", grade or "") if p.strip()]
    return bool(parts) and sum(NO_EVIDENCE in p for p in parts) * 2 > len(parts)


def _counts_as_negative(text: str) -> bool:
    """반대 근거로 셀 수 있는가 — "추론 단독" 만 뺀다 (evaluator 의 추론 단독과 같은 정의, #115).

    - "반대 근거 확보 실패 [추론]" (stakeholder 재작업 소진 표시) → 안 셈
    - "… [https://…][추론]" (도메인: 웹 출처 + 판단) → 셈. 출처 태그가 있으면 [추론] 이 붙어도 근거가 있다
    - "… [추론]" 만 (출처 없음) → 안 셈
    같은 문장이 두 번 들어와도 1건 (호출부에서 set)
    """
    if "확보 실패" in text:
        return False
    if SOURCE_TAG_RE.search(text):
        return True
    return not text.rstrip().endswith("[추론]")


def _fail(gap: str, hint: str, reason: str) -> dict:
    return {"rule": "fail", "judge": None, "gap": gap, "hint_query": hint, "reason": reason}


def _hint(worker: str, tech: str, gap: str) -> str:
    if worker == "tech_research":
        return _TECH_Q.get(gap, _TECH_Q["evidence"]).format(tech=tech)
    if gap == "negatives":
        return f"{tech} 한계 단점 비판 {_KW[worker]}"
    if gap == "source_bias":
        return f"{tech} {_KW[worker]} 독립 분석 기사 보고서"
    if gap == "counter_example":
        return f"{tech} 스마트폰 온디바이스 한계 반례 메모리 대역폭 전력"
    return f"{tech} {_KW[worker]}"


def rule_check(worker: str, tech: str, payload: dict | None, node_status: dict[str, str]) -> dict:
    """셀 하나의 결정론 판정 → CellVerdict (rule pass 면 judge 는 아직 None)."""
    hint = lambda gap: _hint(worker, tech, gap)  # noqa: E731
    if node_status.get(worker) == "failed":
        return _fail("failed", hint("evidence"), f"{worker} 워커 실패 — 실패 기록은 근거로 보지 않는다")
    if payload is None:
        return _fail("missing", hint("evidence"), "미수집")

    evidence = [e for e in payload.get("evidence") or [] if e.get("tag") in GROUNDED_TAGS]
    if worker == "tech_research":
        overview = payload.get("overview", "")
        if overview.startswith(NO_EVIDENCE) or not overview.strip():
            return _fail("no_evidence", hint("evidence"), f"개요가 비어 있거나 근거 없음: {overview[:60]}")
        if len(evidence) < MIN_EVIDENCE:
            return _fail("evidence", hint("evidence"), f"근거 {len(evidence)}건 < {MIN_EVIDENCE}")
        if not payload.get("numbers"):
            return _fail("numbers", hint("numbers"), "실험 수치 없음")
        if not payload.get("limitations"):
            return _fail("limitations", hint("limitations"), "한계(반대 근거) 없음")
        # 기술 조사는 그 기술의 논문 1편이 코퍼스라 출처 다양성은 보지 않는다
        return {"rule": "pass", "judge": None, "gap": "", "hint_query": "", "reason": f"근거 {len(evidence)}건"}

    if payload.get("rationale", "").startswith("워커 실패"):            # safe fallback (node_status 이전 형식)
        return _fail("failed", hint("evidence"), payload["rationale"][:80])
    if _mostly_no_evidence(payload.get("grade", "")):
        return _fail("no_evidence", hint("evidence"), f"등급 과반이 근거 없음: {payload.get('grade', '')}")
    if len(evidence) < MIN_EVIDENCE:
        return _fail("evidence", hint("evidence"), f"근거 {len(evidence)}건 < {MIN_EVIDENCE}")
    neg = len({str(n).strip() for n in payload.get("negatives") or [] if _counts_as_negative(str(n))})
    if neg < MIN_NEGATIVES:
        return _fail("negatives", hint("negatives"), f"반대 근거 {neg}건 < {MIN_NEGATIVES}")
    n_src, top, share = _source_bias(evidence)
    if worker in WEB_PERSPECTIVES:
        if n_src < MIN_SOURCES:
            return _fail("source_bias", hint("source_bias"), f"출처 {n_src}곳 < {MIN_SOURCES}")
        if share > MAX_SOURCE_SHARE:
            return _fail("source_bias", hint("source_bias"), f"출처 편중 {top} {share:.0%} — 과반")
    else:                                                          # domain
        web = sum(e.get("tag") == "웹" for e in evidence)
        if web < MIN_WEB_COUNTER:
            return _fail("counter_example", hint("counter_example"), f"웹 근거 {web}건 < {MIN_WEB_COUNTER} (웹 반례 필요)")
    return {"rule": "pass", "judge": None, "gap": "", "hint_query": "",
            "reason": f"근거 {len(evidence)}건 · 반대 {neg}건 · 출처 {n_src}곳(최다 {share:.0%})"}


def check_rules(state: dict) -> dict[str, dict]:
    """수집된 모든 셀의 결정론 판정. 미수집 셀은 빼고 돌려준다 (미수집은 게이트가 State 에서 직접 본다)."""
    node_status = state.get("node_status") or {}
    out = {}
    for worker, key in PAYLOAD.items():
        data = state.get(key) or {}
        for tech in TECHS:
            if tech in data:
                out[f"{worker}:{tech}"] = rule_check(worker, tech, data[tech], node_status)
    return out


# ── LLM Judge 층 ─────────────────────────────────────────

def _render_content(worker: str, payload: dict) -> str:
    ev = payload.get("evidence") or []
    lines = []
    if worker == "tech_research":
        lines += [f"개요: {payload.get('overview', '')}", f"메커니즘: {payload.get('mechanism', '')}"]
        lines += [f"수치: {x}" for x in payload.get("numbers") or []]
        lines += [f"한계: {x}" for x in payload.get("limitations") or []]
    else:
        lines += [f"등급: {payload.get('grade', '')}", f"근거 요약: {payload.get('rationale', '')}"]
        if payload.get("verdict"):
            lines.append(f"판정: {payload['verdict']}")
        lines += [f"긍정: {x}" for x in payload.get("positives") or []]
        lines += [f"반대: {x}" for x in payload.get("negatives") or []]
    lines.append("근거 문장:")
    for e in ev[:MAX_JUDGE_EVIDENCE]:
        page = f" p.{e['page']}" if e.get("page") else ""
        lines.append(f"- [{e.get('tag', '')}{page}] {e.get('claim', '')}  ({source_key(e.get('ref', ''))})")
    if len(ev) > MAX_JUDGE_EVIDENCE:
        lines.append(f"- … 외 {len(ev) - MAX_JUDGE_EVIDENCE}건")
    return "\n".join(lines)


def judge_cell(worker: str, tech: str, payload: dict) -> SufficiencyJudgment:
    prompt = render_prompt("sufficiency_judge", tech=tech, perspective=PERSPECTIVE[worker],
                           content=_render_content(worker, payload))
    return llm("judge").with_structured_output(SufficiencyJudgment).invoke(prompt)


def _apply_judgment(rule: dict, j: SufficiencyJudgment) -> dict:
    return {**rule, "judge": "sufficient" if j.sufficient else "insufficient",
            "gap": "" if j.sufficient else j.gap, "hint_query": "" if j.sufficient else j.hint_query,
            "reason": f"{rule['reason']} · Judge: {j.reason}"}


def cells_to_judge(state: dict, rules: dict[str, dict]) -> list[str]:
    """이번에 (재)판정할 셀 — 판정이 없는 셀 + 방금 실행된 워커의 셀 (재작업이면 그 기술만)."""
    prev = state.get("sufficiency") or {}
    just_ran = state.get("next") or ""
    rr = state.get("rework_request") or {}
    out = []
    for cell in rules:
        worker, _, tech = cell.partition(":")
        if cell not in prev:
            out.append(cell)
        elif worker == just_ran and (rr.get("worker") != worker or rr.get("tech") == tech):
            out.append(cell)
    return out


def assess(state: dict) -> dict:
    """노드. 판정만 기록한다 — next · retry 는 쓰지 않는다 (게이트 전용)."""
    rules = check_rules(state)
    out, calls = {}, 0
    for cell in cells_to_judge(state, rules):
        v = rules[cell]
        if v["rule"] == "pass":
            worker, _, tech = cell.partition(":")
            try:
                v = _apply_judgment(v, judge_cell(worker, tech, state[PAYLOAD[worker]][tech]))
            except Exception as e:  # noqa: BLE001 — Judge 실패는 규칙 판정으로 남기고 사유에 적는다
                print(f"[assess] {cell} Judge 실패 → 규칙 판정만: {type(e).__name__}: {e}", file=sys.stderr)
                v = {**v, "reason": f"{v['reason']} · Judge 실패({type(e).__name__}) — 규칙 판정만"}
            calls += 1
        out[cell] = v
    return {"sufficiency": out, "llm_calls": calls}


def is_sufficient(v: dict[str, Any] | None) -> bool:
    """게이트 · 보고서 한계점용 — 규칙 통과이고 Judge 가 부족이라 하지 않았으면 충분."""
    return bool(v) and v.get("rule") == "pass" and v.get("judge") != "insufficient"
