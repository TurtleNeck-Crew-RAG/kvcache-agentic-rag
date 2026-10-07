"""최종 보고서 품질 평가 규칙층 — Groundedness · 중립성 · 편향 · 커버리지.

보고서 뒤에서 실행해 ``eval_result`` 만 기록한다. 이 모듈은 LLM을 호출하지 않는다.
의미 기반 근거성·중립성 판정은 후속 LLM Judge가 이 결과 위에 덧붙인다.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

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


def _analysis_lines(body: str) -> list[str]:
    """평가 판단이 쓰이는 SUMMARY·3·4·5장만 반환한다."""
    lines: list[str] = []
    include = False
    fenced = False
    for raw in body.splitlines():
        if raw.strip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        if raw.startswith("## "):
            title = raw[3:].strip()
            include = title == "SUMMARY" or title.startswith(("3.", "4.", "5."))
            continue
        if include:
            lines.append(raw)
    return lines


def claim_units(markdown: str) -> list[str]:
    """출처 태그 비율의 분모가 되는 판단 문장을 재현 가능하게 추출한다."""
    body, _ = _split_reference(markdown)
    units: list[str] = []
    for raw in _analysis_lines(body):
        line = raw.strip()
        if not line or line.startswith(("#", "|", "<!--")):
            continue
        line = re.sub(r"^(?:[-*+]\s+|\d+\.\s+)", "", line).strip()
        plain = re.sub(r"[*_]", "", INLINE_CODE_RE.sub("", line)).strip()
        if not plain or plain.endswith(":") or plain.startswith(("confidence:", "판정:", "긍정:", "부정:")):
            continue
        if TECH_HEADER_RE.match(line) or "공개 정보 기반 추정이다" in plain:
            continue
        parts = [part.strip() for part in SENTENCE_RE.split(line) if len(part.strip()) >= 8]
        units.extend(parts or [line])
    return units


def _tag_stats(markdown: str) -> tuple[int, int, int, float, float]:
    units = claim_units(markdown)
    tagged = inference_only = 0
    for unit in units:
        clean = INLINE_CODE_RE.sub("", unit)
        kinds = {"논문" if tag.startswith("p.") else tag for tag in TAG_RE.findall(clean)}
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
    return {"eval_result": evaluate(markdown, state), "llm_calls": 0}
