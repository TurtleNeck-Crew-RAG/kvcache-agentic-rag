"""최종 보고서 품질 평가 — Groundedness · 중립성 · 편향 · 커버리지.  [소유: C 민영은]

보고서 뒤에서 실행해 ``eval_result`` 만 기록한다. 다음 경로는 Supervisor 게이트가 정한다.
1층 규칙(LLM 없음)을 먼저 돌리고, 규칙을 통과한 항목만 2층 LLM Judge 가 문맥으로 다시 본다.
- Groundedness: 규칙 통과 시 주요 주장 샘플 ↔ State evidence 의미 대응
- 중립성: 규칙 통과 시 비교·추천 단서가 있는 문장만 우열 판정 문맥인지
Judge 호출은 최대 1회이고 ``llm_calls`` 에 반영한다.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal, NamedTuple
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents._common import TECHS, llm, load_prompt
from graph.state import EvalItem, EvalResult, GraphState

MIN_TAGGED_RATIO = 0.90
MAX_INFERENCE_RATIO = 0.10
MAX_WEB_SOURCE_SHARE = 0.40

TAG_RE = re.compile(r"\[(논문|웹|추론|p\.\d)[^\]]*\]")
URL_RE = re.compile(r"https?://[^\s\]\)>]+")
ARXIV_RE = re.compile(r"\b\d{4}\.\d{4,5}\b")
INLINE_CODE_RE = re.compile(r"`[^`]*`")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
TECH_HEADER_RE = re.compile(r"^(?:-\s*)?\*\*(?:KIVI|InfiniGen)\*\*\s*[—:-]")
DIRECT_RECOMMENDATION_RE = re.compile(
    r"(?:KIVI|InfiniGen).{0,30}(?:선택해야|권장(?:한다|된다)|더\s+낫다|더\s+우수하다)",
    re.IGNORECASE,
)

REQUIRED_HEADINGS = {
    "summary": re.compile(r"^## SUMMARY\s*$", re.MULTILINE),
    "technical_maturity": re.compile(r"^### 4\.1\b.*$", re.MULTILINE),
    "market": re.compile(r"^### 4\.2\b.*$", re.MULTILINE),
    "stakeholder": re.compile(r"^### 4\.3\b.*$", re.MULTILINE),
    "domain": re.compile(r"^### 4\.4\b.*$", re.MULTILINE),
    "reference": re.compile(r"^## REFERENCE\s*$", re.MULTILINE),
}
PERSPECTIVE_HEADINGS = ("4.1", "4.2", "4.3", "4.4")

# 보고서 절 → 원천 근거를 만든 워커와 그 State 키. target "{worker}:{tech}" 는 Supervisor PAYLOAD 와 같은 이름을 쓴다.
SECTION_WORKERS = {"4.1": "tech_research", "4.2": "market", "4.3": "stakeholder", "4.4": "domain"}
WORKER_KEYS = {
    "tech_research": "tech_summary",
    "market": "market_eval",
    "stakeholder": "stakeholder_eval",
    "domain": "domain_eval",
}

MAX_JUDGE_CLAIMS = 12
MAX_NEUTRALITY_SUSPECTS = 10
MAX_EVIDENCE_PER_CLAIM = 8
MAX_EVIDENCE_CHARS = 300
NEUTRALITY_CUE_RE = re.compile(
    r"더\s*(?:낫|우수|좋|뛰어|적합|유리|효과적|바람직)|우수|우월|열등|뒤처|앞선|앞서|최선|최적(?!화)|"
    r"권장|권고|추천|선택해야|채택해야|도입해야|바람직|유리하다"
)


class Claim(NamedTuple):
    text: str
    worker: str | None   # 이 문장이 놓인 절의 원천 워커 (3장·4.x). SUMMARY · 5장은 None
    tech: str | None


def _item(passed: bool, score: float, reason: str) -> EvalItem:
    return {"passed": passed, "score": round(max(0.0, min(1.0, score)), 2), "reason": reason}


def _failed_result(reason: str) -> dict:
    items = {
        name: _item(False, 0.0, reason)
        for name in ("groundedness", "neutrality", "bias", "coverage")
    }
    result: EvalResult = {
        "passed": False,
        "items": items,  # type: ignore[typeddict-item]
        "targets": ["report"],
        "feedback": reason,
    }
    return {"eval_result": result, "llm_calls": 0}


def _split_reference(markdown: str) -> tuple[str, str]:
    match = REQUIRED_HEADINGS["reference"].search(markdown)
    if not match:
        return markdown, ""
    return markdown[:match.start()], markdown[match.end():]


def _mentioned_tech(text: str) -> str | None:
    found = [tech for tech in TECHS if tech in text]
    return found[0] if len(found) == 1 else None


def claim_records(markdown: str) -> list[Claim]:
    """SUMMARY·3·4·5장의 판단 문장을 절(워커)·기술 문맥과 함께 재현 가능하게 추출한다."""
    body, _ = _split_reference(markdown)
    records: list[Claim] = []
    include = fenced = False
    worker: str | None = None
    tech: str | None = None
    for raw in body.splitlines():
        if raw.strip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        if raw.startswith("## "):
            title = raw[3:].strip()
            include = title == "SUMMARY" or title.startswith(("3.", "4.", "5."))
            worker, tech = ("tech_research" if title.startswith("3.") else None), None
            continue
        if not include:
            continue
        if raw.startswith("### "):
            title = raw[4:].strip()
            if title in TECHS:
                tech = title
            else:
                worker, tech = SECTION_WORKERS.get(title.split()[0] if title else ""), None
            continue
        line = raw.strip()
        if not line or line.startswith(("#", "|", "<!--")):
            continue
        line = re.sub(r"^(?:[-*+]\s+|\d+\.\s+)", "", line).strip()
        plain = re.sub(r"[*_]", "", INLINE_CODE_RE.sub("", line)).strip()
        if TECH_HEADER_RE.match(line):
            tech = _mentioned_tech(line)
            continue
        if not plain or plain.endswith(":") or plain.startswith(("confidence:", "판정:", "긍정:", "부정:")):
            continue
        if "공개 정보 기반 추정이다" in plain:
            continue
        parts = [part.strip() for part in SENTENCE_RE.split(line) if len(part.strip()) >= 8]
        records.extend(Claim(part, worker, tech or _mentioned_tech(part)) for part in parts or [line])
    return records


def claim_units(markdown: str) -> list[str]:
    """출처 태그 비율의 분모가 되는 판단 문장."""
    return [record.text for record in claim_records(markdown)]


def _source_kinds(unit: str) -> set[str]:
    clean = INLINE_CODE_RE.sub("", unit)
    return {"논문" if tag.startswith("p.") else tag for tag in TAG_RE.findall(clean)}


def _tag_stats(markdown: str) -> tuple[int, int, int, float, float]:
    units = claim_units(markdown)
    tagged = inference_only = 0
    for unit in units:
        kinds = _source_kinds(unit)
        if kinds:
            tagged += 1
        if kinds == {"추론"}:
            inference_only += 1
    total = len(units)
    return (
        total,
        tagged,
        inference_only,
        tagged / total if total else 0.0,
        inference_only / total if total else 0.0,
    )


def _clean_urls(text: str) -> set[str]:
    return {url.rstrip(".,;") for url in URL_RE.findall(text)}


def _citation_correspondence(markdown: str) -> tuple[bool, str]:
    body, reference = _split_reference(markdown)
    if not reference.strip():
        return False, "REFERENCE 없음"

    body_urls, reference_urls = _clean_urls(body), _clean_urls(reference)
    body_ids, reference_ids = set(ARXIV_RE.findall(body)), set(ARXIV_RE.findall(reference))
    missing = sorted(body_urls - reference_urls) + sorted(body_ids - reference_ids)
    uncited_urls = sorted(reference_urls - body_urls)

    # 페이지형 논문 태그는 보고서 렌더러가 코퍼스 논문 인용으로 취급한다.
    generic_paper_tag = bool(re.search(r"\[(?:논문\s+)?p\.\d", body))
    uncited_ids = [] if generic_paper_tag else sorted(reference_ids - body_ids)
    if "[논문" in body and not reference_ids:
        missing.append("논문 REFERENCE")

    problems = []
    if missing:
        problems.append("REFERENCE 누락: " + ", ".join(missing))
    if uncited_urls or uncited_ids:
        problems.append("본문 미인용 REFERENCE: " + ", ".join(uncited_urls + uncited_ids))
    return not problems, "; ".join(problems) if problems else "본문 인용과 REFERENCE 대응"


def groundedness(markdown: str) -> EvalItem:
    total, tagged, inference_only, tagged_ratio, inference_ratio = _tag_stats(markdown)
    refs_ok, refs_reason = _citation_correspondence(markdown)
    passed = bool(total) and tagged_ratio >= MIN_TAGGED_RATIO and inference_ratio <= MAX_INFERENCE_RATIO and refs_ok
    score = min(tagged_ratio, 1.0 - inference_ratio, 1.0 if refs_ok else 0.0)
    reason = (
        f"판단 문장 태그 {tagged}/{total}({tagged_ratio:.0%}) · "
        f"추론 단독 {inference_only}/{total}({inference_ratio:.0%}) · {refs_reason}"
    )
    return _item(passed, score, reason)


def neutrality(markdown: str) -> EvalItem:
    """명백한 직접 권고만 규칙으로 거른다. 문맥 판정은 후속 LLM Judge가 맡는다."""
    body, _ = _split_reference(markdown)
    violations = DIRECT_RECOMMENDATION_RE.findall(body)
    if violations:
        return _item(False, 0.0, f"명시적 기술 추천·우열 표현 {len(violations)}건")
    return _item(True, 1.0, "명시적 기술 추천 표현 없음 — 문맥 판정은 LLM Judge 대상")


def _source_key(ref: str) -> str:
    if not ref.startswith(("http://", "https://")):
        return ""
    return urlparse(ref).netloc.removeprefix("www.").lower()


def bias(state: GraphState) -> EvalItem:
    missing_negatives: list[str] = []
    web_sources: list[str] = []

    for tech in ("KIVI", "InfiniGen"):
        tech_summary = (state.get("tech_summary") or {}).get(tech) or {}
        if not tech_summary.get("limitations"):
            missing_negatives.append(f"tech_research:{tech}")
        for worker, key in (("market", "market_eval"), ("stakeholder", "stakeholder_eval"), ("domain", "domain_eval")):
            value = (state.get(key) or {}).get(tech) or {}
            if not value.get("negatives"):
                missing_negatives.append(f"{worker}:{tech}")
            web_sources.extend(
                source for evidence in value.get("evidence") or []
                if evidence.get("tag") == "웹" and (source := _source_key(str(evidence.get("ref", ""))))
            )

    counts = Counter(web_sources)
    top_source, top_count = counts.most_common(1)[0] if counts else ("", 0)
    top_share = top_count / len(web_sources) if web_sources else 1.0
    passed = not missing_negatives and bool(web_sources) and top_share <= MAX_WEB_SOURCE_SHARE
    negative_score = 1.0 - len(missing_negatives) / 8
    source_score = 1.0 - top_share if web_sources else 0.0
    details = []
    if missing_negatives:
        details.append("반대 근거 없음: " + ", ".join(missing_negatives))
    if not web_sources:
        details.append("웹 근거 없음")
    elif top_share > MAX_WEB_SOURCE_SHARE:
        details.append(f"웹 출처 편중 {top_source} {top_share:.0%} > {MAX_WEB_SOURCE_SHARE:.0%}")
    reason = "; ".join(details) if details else f"8셀 반대 근거 존재 · 최다 웹 출처 {top_source} {top_share:.0%}"
    return _item(passed, (negative_score + source_score) / 2, reason)


def _perspective_section(markdown: str, heading: str) -> str:
    match = re.search(rf"^### {re.escape(heading)}\b.*$", markdown, re.MULTILINE)
    if not match:
        return ""
    tail = markdown[match.end():]
    end = re.search(r"^(?:### 4\.[1-4]\b|## )", tail, re.MULTILINE)
    return tail[:end.start()] if end else tail


def coverage(markdown: str) -> EvalItem:
    missing = [name for name, pattern in REQUIRED_HEADINGS.items() if not pattern.search(markdown)]
    for heading in PERSPECTIVE_HEADINGS:
        section = _perspective_section(markdown, heading)
        for tech in ("KIVI", "InfiniGen"):
            if tech not in section:
                missing.append(f"{heading}:{tech}")
    reference = _split_reference(markdown)[1]
    if reference and not re.search(r"^-\s+\S", reference, re.MULTILINE):
        missing.append("reference:item")
    total_checks = len(REQUIRED_HEADINGS) + len(PERSPECTIVE_HEADINGS) * 2 + 1
    score = 1.0 - len(set(missing)) / total_checks
    return _item(not missing, score, "필수 구성 모두 존재" if not missing else "누락: " + ", ".join(missing))


def evaluate(markdown: str, state: GraphState) -> EvalResult:
    """1층 규칙 평가. LLM 없이 네 항목을 모두 채운다."""
    items = {
        "groundedness": groundedness(markdown),
        "neutrality": neutrality(markdown),
        "bias": bias(state),
        "coverage": coverage(markdown),
    }
    failed = [name for name, item in items.items() if not item["passed"]]
    return {
        "passed": not failed,
        "items": items,  # type: ignore[typeddict-item]
        "targets": ["report"] if failed else [],
        "feedback": "\n".join(f"{name}: {items[name]['reason']}" for name in failed),
    }


# ── 2층 LLM Judge ────────────────────────────────────────

class ClaimJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="검증 대상 주장 id (C1, C2, ...)")
    problem: Literal["none", "overstated", "unsupported"] = Field(
        description="none=근거와 대응 · overstated=근거보다 강한 주장 · unsupported=대응하는 근거 없음",
    )
    feedback: str = Field(description="문제가 있으면 무엇을 어떻게 고칠지 한 문장. none 이면 빈 문자열")


class NeutralityJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="검증 대상 문장 id (N1, N2, ...)")
    violation: bool = Field(description="특정 기술의 우열 단정 또는 선택·도입 권고이면 true")
    feedback: str = Field(description="위반이면 중립적으로 고쳐 쓸 방향 한 문장. 아니면 빈 문자열")


class EvaluatorJudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[ClaimJudgement]
    neutrality: list[NeutralityJudgement]

    @model_validator(mode="after")
    def problems_have_feedback(self) -> EvaluatorJudgeOutput:
        if any(c.problem != "none" and not c.feedback.strip() for c in self.claims) or any(
            n.violation and not n.feedback.strip() for n in self.neutrality
        ):
            raise ValueError("문제로 판정한 항목에는 수정 가능한 feedback 이 필요합니다")
        return self


def judge_claims(markdown: str) -> list[Claim]:
    """근거 태그([논문]·[웹])가 붙은 판단 문장 중 셀(절 × 기술)마다 고르게, 수치가 있는 문장부터 표본을 뽑는다."""
    groups: dict[tuple[str | None, str | None], list[Claim]] = {}
    for record in claim_records(markdown):
        content = TAG_RE.sub("", record.text).strip(" .()")
        if len(content) >= 8 and _source_kinds(record.text) - {"추론"}:   # 태그만 있는 줄은 주장이 아니다
            groups.setdefault((record.worker, record.tech), []).append(record)
    for group in groups.values():
        group.sort(key=lambda c: not re.search(r"\d", TAG_RE.sub("", c.text)))
    picked: list[Claim] = []
    while len(picked) < MAX_JUDGE_CLAIMS and any(groups.values()):
        for group in groups.values():
            if group and len(picked) < MAX_JUDGE_CLAIMS:
                picked.append(group.pop(0))
    return picked


def neutrality_suspects(markdown: str) -> list[str]:
    """비교·추천 단서가 있는 판단 문장만 Judge 에 넘긴다. 위반 여부는 단서가 아니라 문맥으로 판정한다."""
    seen: list[str] = []
    for record in claim_records(markdown):
        if NEUTRALITY_CUE_RE.search(TAG_RE.sub("", record.text)) and record.text not in seen:
            seen.append(record.text)
    return seen[:MAX_NEUTRALITY_SUSPECTS]


def _clip(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= MAX_EVIDENCE_CHARS else text[:MAX_EVIDENCE_CHARS - 1] + "…"


def _evidence_text(evidence: dict) -> str:
    page = f" p.{evidence['page']}" if evidence.get("page") else ""
    return _clip(f"[{evidence.get('tag', '')} {evidence.get('ref', '')}{page}] {evidence.get('claim', '')}")


def _evidence_for(claim: Claim, state: GraphState) -> list[str]:
    """주장과 같은 셀(워커·기술)의 State 근거. 본문에 적힌 URL 과 같은 ref 는 셀과 무관하게 앞에 둔다."""
    urls = _clean_urls(claim.text)
    workers = [claim.worker] if claim.worker else list(WORKER_KEYS)
    techs = [claim.tech] if claim.tech else list(TECHS)
    matched: list[str] = []
    related: list[str] = []
    for worker, key in WORKER_KEYS.items():
        for tech in TECHS:
            value = (state.get(key) or {}).get(tech) or {}
            in_cell = worker in workers and tech in techs
            for evidence in value.get("evidence") or []:
                if str(evidence.get("ref", "")) in urls:
                    matched.append(_evidence_text(evidence))
                elif in_cell:
                    related.append(_evidence_text(evidence))
            if in_cell:
                for field in ("numbers", "limitations", "apply_conditions", "positives", "negatives"):
                    related.extend(_clip(text) for text in value.get(field) or [])
    unique = list(dict.fromkeys(matched + related))
    return unique[:MAX_EVIDENCE_PER_CLAIM]


def _claim_target(claim: Claim) -> str:
    return f"{claim.worker}:{claim.tech}" if claim.worker in WORKER_KEYS and claim.tech in TECHS else "report"


def _render_judge_prompt(claims: list[Claim], suspects: list[str], state: GraphState) -> str:
    payload = {
        "claims": [
            {"id": f"C{i}", "text": claim.text, "evidence": _evidence_for(claim, state)}
            for i, claim in enumerate(claims, 1)
        ],
        "neutrality": [{"id": f"N{i}", "text": text} for i, text in enumerate(suspects, 1)],
    }
    return load_prompt("evaluator") + "\n\n## 검증 대상\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _invoke_judge(prompt: str) -> EvaluatorJudgeOutput:
    result = llm("judge").with_structured_output(EvaluatorJudgeOutput).invoke(prompt)
    return EvaluatorJudgeOutput.model_validate(result) if isinstance(result, dict) else result


def _quote(text: str) -> str:
    return "「" + _clip(text) + "」"


def judge(markdown: str, state: GraphState, rule_result: EvalResult) -> tuple[EvalResult, int]:
    """규칙을 통과한 groundedness · neutrality 만 LLM 으로 재판정한다. (갱신된 결과, LLM 호출 수)"""
    items = dict(rule_result["items"])
    claims = judge_claims(markdown) if items["groundedness"]["passed"] else []
    suspects = neutrality_suspects(markdown) if items["neutrality"]["passed"] else []
    if not claims and not suspects:
        return rule_result, 0

    output = _invoke_judge(_render_judge_prompt(claims, suspects, state))
    claim_by_id = {f"C{i}": claim for i, claim in enumerate(claims, 1)}
    suspect_by_id = {f"N{i}": text for i, text in enumerate(suspects, 1)}
    targets: list[str] = []
    notes: list[str] = []

    weak = [(claim_by_id[j.id], j) for j in output.claims if j.id in claim_by_id and j.problem != "none"]
    if claims:
        item = items["groundedness"]
        if weak:
            for claim, j in weak:
                # 표현이 근거보다 강하면 보고서 수정, 근거 자체가 없으면 그 셀의 워커 재조사
                target = _claim_target(claim) if j.problem == "unsupported" else "report"
                targets.append(target)
                label = "근거보다 강한 주장" if j.problem == "overstated" else "근거 없는 단정"
                notes.append(f"groundedness: {_quote(claim.text)} {label} — {j.feedback} (→ {target})")
            score = min(item["score"], 1.0 - len(weak) / len(claims))
            items["groundedness"] = _item(
                False, score, f"{item['reason']} · Judge 근거 불일치 {len(weak)}/{len(claims)}",
            )
        else:
            items["groundedness"] = _item(True, item["score"], f"{item['reason']} · Judge 주장 {len(claims)}건 근거 대응")

    violations = [(suspect_by_id[j.id], j) for j in output.neutrality if j.id in suspect_by_id and j.violation]
    if suspects:
        if violations:
            targets.append("report")
            notes.extend(f"neutrality: {_quote(text)} 우열·추천 문맥 — {j.feedback} (→ report)" for text, j in violations)
            items["neutrality"] = _item(
                False, 1.0 - len(violations) / len(suspects),
                f"Judge 우열·추천 문맥 {len(violations)}/{len(suspects)}건",
            )
        else:
            items["neutrality"] = _item(True, 1.0, f"비교·추천 단서 문장 {len(suspects)}건 모두 중립 문맥 (Judge)")

    # 재조사 target 을 앞에 둔다 — 워커 재실행은 synthesis · report 를 비워 보고서까지 다시 흐른다
    ordered = [t for t in [*targets, *rule_result["targets"]] if t != "report"]
    if "report" in targets or "report" in rule_result["targets"]:
        ordered.append("report")
    feedback = "\n".join(line for line in (rule_result["feedback"], *notes) if line)
    result: EvalResult = {
        "passed": all(item["passed"] for item in items.values()),
        "items": items,  # type: ignore[typeddict-item]
        "targets": list(dict.fromkeys(ordered)),
        "feedback": feedback,
    }
    return result, 1


def run(state: GraphState) -> dict:
    uri = state.get("report_uri")
    if not uri:
        return _failed_result("report_uri 없음")
    path = Path(uri)
    if not path.is_file():
        return _failed_result(f"보고서 파일 없음: {uri}")
    try:
        markdown = path.read_text(encoding="utf-8")
    except OSError as exc:
        return _failed_result(f"보고서 읽기 실패: {type(exc).__name__}")
    if not markdown.strip():
        return _failed_result("보고서가 비어 있음")
    result, calls = judge(markdown, state, evaluate(markdown, state))
    return {"eval_result": result, "llm_calls": calls}
