"""보고서 렌더러 — LLM 없이 State 만으로 만드는 장(章)들.

설계서 6장. LLM 을 쓰지 않는 부분을 여기 모아 테스트 가능하게 둔다:
  2 기술 선정(selection.yaml) · 4 관점별 평가(*_eval) · 6 한계점(수치 계산) · REFERENCE(인용된 citations 만)
LLM 이 쓰는 장(SUMMARY · 1 · 3 · 5)은 agents/report.py.
이 모듈은 langchain 을 import 하지 않는다 — tests 가 CI(langgraph + pytest 만)에서 돌아야 한다.
"""
from __future__ import annotations

import re
from typing import Any

TECHS = ("KIVI", "InfiniGen")
TAG_RE = re.compile(r"\[(논문|웹|추론|p\.\d)[^\]]*\]")     # [논문 p.N] · [웹 URL] · [추론] · [p.N](논문)
FAIL_MARK = "워커 실패"                                        # graph/safe.py fallback 이 남기는 표식

CHAPTERS = (           # (key, 제목) — 이 순서가 목차다. SUMMARY 맨 앞 · REFERENCE 맨 뒤 고정
    ("summary", "SUMMARY"),
    ("background", "1. 분석 배경"),
    ("selection", "2. 기술 선정"),
    ("overview", "3. 기술 개요"),
    ("evaluation", "4. 관점별 평가"),
    ("implications", "5. 시사점"),
    ("limitations", "6. 한계점"),
    ("reference", "REFERENCE"),
)


# ---------- REFERENCE ----------

ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|html|pdf)/(\d{4}\.\d{4,5})")


def arxiv_id_from_url(url: str) -> str | None:
    """https://arxiv.org/abs/2406.19707 · /html/2402.02750v2 · /pdf/2604.05012 → 버전 뗀 arXiv id."""
    m = ARXIV_RE.search(url or "")
    return m.group(1) if m else None


def format_authors(names: list[str], max_names: int = 3) -> str:
    """['Keivan Alizadeh', 'Iman Mirzadeh'] → 'Alizadeh, K., Mirzadeh, I.' — 설계서 6장 예시 표기. 4명 이상은 et al."""
    out = []
    for n in names[:max_names]:
        parts = n.strip().split()
        if not parts:
            continue
        last, initials = parts[-1], "".join(p[0] + "." for p in parts[:-1])
        out.append(f"{last}, {initials}" if initials else last)
    s = ", ".join(out)
    return s + " et al." if len(names) > max_names else s


def _selected_papers(selected: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """config/selection.yaml 의 sw/hw → {arxiv id: 메타}. 선정 논문 2편은 저자·학회가 여기 있다."""
    out: dict[str, dict[str, Any]] = {}
    for side in ("sw", "hw"):
        p = selected.get(side) or {}
        if p.get("arxiv"):
            out[str(p["arxiv"])] = p
    return out


def normalize_citations(citations: list[dict[str, Any]], selected: dict[str, Any] | None = None,
                        meta: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """웹검색이 논문 페이지(arxiv.org)를 긁어 온 `웹` citation 을 `논문` citation 으로 바꾸고 같은 논문을 하나로 합친다.

    - 선정 논문 2편(selection.yaml)이면 그 저자·학회를 쓴다
    - 그 외 arXiv id 는 meta(arXiv API 캐시: authors · published · title)가 있으면 저자·연도를 채우고 학회는 *arXiv*
    - meta 도 없으면 제목·id 만으로 논문 항목을 만든다 (저자 미확인)
    같은 arXiv id 가 여러 번 나오면 학회 정보가 있는 쪽 하나만 남긴다.
    """
    sel = _selected_papers(selected or {})
    meta = meta or {}
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    others: list[dict[str, Any]] = []
    for ref in citations:
        aid = None
        if ref.get("type") == "논문":
            aid = _norm_id(str(ref.get("id_or_url") or ""))
            aid = aid if re.fullmatch(r"\d{4}\.\d{4,5}", aid) else None
        elif ref.get("type") == "웹":
            aid = arxiv_id_from_url(str(ref.get("id_or_url") or ""))
        if not aid:
            others.append(ref)
            continue
        cand = dict(ref)
        if cand.get("type") == "웹":                                 # 웹 → 논문 승격
            p = sel.get(aid)
            m = meta.get(aid) or {}
            cand = {
                "type": "논문",
                "authors": (p or {}).get("authors") or (format_authors(m["authors"]) if m.get("authors") else ""),
                "year": (str((p or {}).get("year") or "") or (m.get("published") or "")[:4] or ""),
                "title": (p or {}).get("paper") or m.get("title") or ref.get("title") or "",
                "venue": (p or {}).get("venue") or "arXiv",
                "id_or_url": f"arXiv:{aid}",
                "accessed": ref.get("accessed", ""),
            }
        if aid not in by_id:
            by_id[aid] = cand
            order.append(aid)
        else:                                                      # 더 완전한 쪽을 남긴다
            cur = by_id[aid]
            score = lambda r: (r.get("venue", "arXiv") != "arXiv", bool(r.get("authors")), bool(r.get("year")))  # noqa: E731
            if score(cand) > score(cur):
                by_id[aid] = cand
    return [by_id[a] for a in order] + others

def _norm_id(id_or_url: str) -> str:
    return id_or_url.replace("arXiv:", "").strip()


def is_cited(ref: dict[str, Any], body: str, corpus_ids: set[str] | None = None) -> bool:
    """본문에서 실제로 인용됐는가 (설계서 6장 — 미인용 자료는 제외).

    - arXiv id · URL · 제목이 본문에 있으면 인용
    - RAG 코퍼스 논문(corpus_ids = 선정 2편)은 본문에 `[논문 p.N]`/`[p.N]` 태그가 하나라도 있으면 인용으로 본다 —
      워커가 `[논문 p.7]` 처럼 id 없이 태그를 달기 때문. 풀 밖 논문(웹검색으로 긁어 온 arXiv)에는 이 규칙을 쓰지 않는다
    """
    ident = _norm_id(ref["id_or_url"])
    if ident and ident in body:
        return True
    if ref["title"] and ref["title"] in body:
        return True
    if ref["type"] == "논문" and ident in (corpus_ids or set()) and ("[논문" in body or re.search(r"\[p\.\d", body)):
        return True
    return False


def format_ref(ref: dict[str, Any]) -> str:
    """설계서 6장 REFERENCE 표기 형식."""
    t = ref["type"]
    if t == "논문":
        # 저자(YYYY). 논문제목. *학술지/학회명*, 권(호), 페이지.
        venue = f" *{ref['venue']}*." if ref.get("venue") else ""
        authors = ref.get("authors") or "(저자 미확인)"
        year = ref.get("year") or "연도 미확인"
        return f"{authors}({year}). {ref['title']}.{venue} {ref['id_or_url']}.".replace("..", ".")
    if t == "특허":
        # 출원인(YYYY-MM). *특허명*, 특허번호/공개번호, URL
        return f"{ref['authors']}({ref['year']}). *{ref['title']}*, {ref['id_or_url']}."
    # 기타(웹): 기관명 또는 작성자(YYYY-MM-DD). *제목*. 사이트명, URL
    # Tavily 는 저자·게시일을 주지 않는 경우가 많다 → 저자 미상이면 기관명(GitHub 소유자 / 사이트)으로, 날짜는 게시 연도가 있으면 연도,
    # 없으면 "게시일 미상" 을 적고 접근일은 "접근" 을 붙여 게시일로 오해하지 않게 한다 (설계서 6장 스키마에 게시일 필드가 없음 — 한계점 5 기록)
    author = _web_author(ref)
    year = str(ref.get("year") or "")
    when = year if year and "미상" not in year else "게시일 미상"
    accessed = ref.get("accessed") or ""
    date = f"{when} · {accessed} 접근" if accessed else when
    return f"{author}({date}). *{ref['title']}*. {ref['venue']}, {ref['id_or_url']}"


def _web_author(ref: dict[str, Any]) -> str:
    author = str(ref.get("authors") or "").strip()
    if author and "미상" not in author:
        return author
    url = str(ref.get("id_or_url") or "")
    m = re.match(r"https?://(?:www\.)?github\.com/([^/]+)/", url)
    if m:
        return m.group(1)                          # GitHub 은 소유자를 작성자로 (설계서 예시 jy-yuan)
    venue = str(ref.get("venue") or "")
    return venue.removeprefix("www.") or "작성자 미상"


def render_reference(citations: list[dict[str, Any]], body: str, corpus_ids: set[str] | None = None) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for ref in citations:
        key = _norm_id(ref["id_or_url"]) or ref["title"]
        if key in seen or not is_cited(ref, body, corpus_ids):
            continue
        seen.add(key)
        lines.append(f"- {format_ref(ref)}")
    return "\n".join(lines) if lines else "- (본문에 인용된 자료 없음)"


# ---------- 2. 기술 선정 (config/selection.yaml → State.selected) ----------

def render_selection(selected: dict[str, Any]) -> str:
    out = ["**선정 기준** (설계서 1.1)", ""]
    for c in selected.get("criteria", []):
        out.append(f"- **{c['label']}** — {c['question']}")
    out += ["", "**선정 기술**", ""]
    for side, label in (("sw", "SW"), ("hw", "HW")):
        s = selected.get(side, {})
        if not s:
            continue
        out.append(f"- **{label} · {s['name']}** — {s['paper']} ({s.get('venue', '')}, arXiv:{s.get('arxiv', '')}, {s.get('pages', '?')}p)")
        out.append(f"  - {s.get('reason', '').strip()}")
    excluded = selected.get("excluded", [])
    if excluded:
        out += ["", "**검토했으나 제외한 후보**", ""]
        for e in excluded:
            out.append(f"- {e['name']} ({e['side'].upper()}) — {e['reason']}")
    return "\n".join(out)


# ---------- 4. 관점별 평가 (*_eval · trl_estimate) ----------

def _bullets(items: list[str], indent: str = "    ") -> list[str]:
    return [f"{indent}- {x}" for x in items] if items else [f"{indent}- 근거 없음"]


def _eval_block(tech: str, e: dict[str, Any]) -> list[str]:
    """중첩 목록은 4칸 들여쓰기 — python-markdown(PDF 변환)은 2칸 들여쓰기를 목록으로 보지 않아 한 문단으로 뭉쳤다(7회차 PDF 6쪽)."""
    out = [f"- **{tech}** — 등급: {e.get('grade', '근거 없음')}"]
    if e.get("verdict"):
        out.append(f"    - 판정: {e['verdict']}")
    if e.get("rationale"):
        out.append(f"    - 근거: {e['rationale']}")
    out.append("    - 긍정:")
    out += _bullets(e.get("positives", []), "        ")
    out.append("    - 부정:")
    out += _bullets(e.get("negatives", []), "        ")
    if e.get("axes"):
        out.append("    - 3축: " + " · ".join(f"{k} — {v}" for k, v in e["axes"].items()))
    if "confidence" in e:
        out.append(f"    - confidence: {e['confidence']}")
    return out


def render_evaluation(state: dict[str, Any]) -> str:
    out: list[str] = []
    trl = state.get("trl_estimate") or {}
    out.append("### 4.1 기술 성숙도 (TRL — 공개 정보 기반 추정, 기준 시점 명시)")
    out.append("")
    for tech in TECHS:
        t = trl.get(tech)
        if not t:
            out.append(f"- **{tech}** — 근거 없음")
            continue
        out.append(f"- **{tech}** — TRL {t['level']} (기준 시점: {t.get('reference_date', '미기재')})")
        out += _bullets(t.get("basis", []))
    out.append("")
    out.append("TRL 4~6 구간은 수율·성능 수치가 비공개라 정보 공백이 크고, 발표 시점과 채택 사이에 시차가 있다. 위 등급은 공개 정보 기반 추정이다.")
    for num, title, key in (
        ("4.2", "시장", "market_eval"),
        ("4.3", "이해관계자 (찬 · 반)", "stakeholder_eval"),
        ("4.4", "도메인 — 스마트폰 온디바이스 (적합 / 조건부 / 부적합 + 포기한 축)", "domain_eval"),
    ):
        out += ["", f"### {num} {title}", ""]
        evals = state.get(key) or {}
        for tech in TECHS:
            e = evals.get(tech)
            out += _eval_block(tech, e) if e else [f"- **{tech}** — 근거 없음"]
            out.append("")
    return "\n".join(out).rstrip()


# ---------- 6. 한계점 — 수치 계산 ----------

def _tag_kind(t: str) -> str:
    return "논문" if t.startswith("p.") else t


def _tagged_statements(state: dict[str, Any]) -> list[tuple[str, set[str]]]:
    """출처 태그가 붙은 판단 문장을 (문장, {태그 종류}) 로 모은다.

    두 형식을 다 센다 — 문장 안 인라인 태그(`[논문 p.7]` · `[p.2]` · `[추론]`)와
    `Evidence` dict(`{"claim", "tag", ...}` — 워커 실출력은 태그를 여기에 둔다).
    """
    found: list[tuple[str, set[str]]] = []

    def walk(x: Any) -> None:
        if isinstance(x, str):
            tags = {_tag_kind(t) for t in TAG_RE.findall(x)}
            if tags:
                found.append((x, tags))
        elif isinstance(x, dict):
            if "claim" in x and "tag" in x:               # Evidence dict
                found.append((str(x.get("claim", "")), {str(x["tag"])}))
                return
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    for key in ("market_eval", "stakeholder_eval", "domain_eval", "tech_summary", "synthesis", "trl_estimate"):
        walk(state.get(key))
    return found


def is_failed(value: Any) -> bool:
    """graph/safe.py 의 fallback 인가 (rationale · conflicts · basis 에 FAIL_MARK)."""
    return FAIL_MARK in json_dumps(value)


def json_dumps(value: Any) -> str:
    import json
    return json.dumps(value, ensure_ascii=False) if value is not None else ""


def limitation_stats(state: dict[str, Any]) -> dict[str, Any]:
    tagged = _tagged_statements(state)
    stmts = [s for s, _ in tagged]
    inference_only = [s for s, tags in tagged if tags == {"추론"}]
    log = state.get("retrieval_log") or []
    rewritten = [r for r in log if r.get("rewritten")]
    return {
        "tagged_total": len(stmts),
        "inference_only": len(inference_only),
        "inference_ratio": round(len(inference_only) / len(stmts), 2) if stmts else None,
        "retrieval_total": len(log),
        "no_evidence": sum(1 for r in log if r.get("relevance") == "no_evidence"),
        "rewritten": len(rewritten),
        "rewrite_recovered": sum(1 for r in rewritten if r.get("relevance") == "yes"),
        "by_tech_no_evidence": {t: sum(1 for r in log if r.get("tech") == t and r.get("relevance") == "no_evidence") for t in TECHS},
        "neutrality_violations": list((state.get("neutrality") or {}).get("violations") or []),
    }


def render_limitations(state: dict[str, Any], stats: dict[str, Any] | None = None,
                       retrieval_metrics: dict[str, Any] | None = None) -> str:
    """설계서 6장 한계점 6항목. retrieval_metrics = {"hit@4":…, "mrr@4":…, "ragas": {...}} (A 의 rag.evaluate 결과, 없으면 TBD)."""
    st = stats or limitation_stats(state)
    m = retrieval_metrics or {}
    ratio = "계산 불가(태그 문장 없음)" if st["inference_ratio"] is None else f"{st['inference_ratio']:.0%} ({st['inference_only']}/{st['tagged_total']})"
    hit = m.get("hit@4", "TBD")
    mrr = m.get("mrr@4", "TBD")
    mode = f" (검색 구성 `{m['mode']}`, 20문항)" if m.get("mode") else ""
    ragas = m.get("ragas") or {}
    rewrite = (f"재작성 {st['rewritten']}건 중 {st['rewrite_recovered']}건이 관련 문서를 찾았다"
               if st["rewritten"] else "재작성이 발생하지 않았다")
    no_ev = st["no_evidence"]
    by_tech = " · ".join(f"{t} {n}건" for t, n in st["by_tech_no_evidence"].items())
    viol = st["neutrality_violations"]
    viol_line = (f" 중립성 검증에서 반려 상한(2회) 후에도 남은 위반 표현 {len(viol)}건: " + "; ".join(viol)) if viol else ""
    return "\n".join([
        "1. **공개 정보 기반 추정의 한계** — TRL 4~6 구간은 수율·성능 수치가 비공개라 정보 공백이 가장 크다. 4.1 의 등급은 공개 코드·재현·프레임워크 통합 여부만으로 추정한 것이다.",
        "2. **TRL 기준 시점** — arXiv v1 과 학회 게재·가이드 표기 시점이 논문마다 다르다(KIVI: v1 2024-02 / ICML 2024-07, InfiniGen: v1 2024-06 / OSDI 2024-07). 기준 시점에 따라 추정이 달라진다 — 발표 시점과 채택 간 시차의 구체 사례다.",
        "3. **사전 가설과 확증편향 방지 조치** — 사전 가설 *\"온디바이스에서는 SW 압축이 더 적합할 것\"* 을 명시하고 장치 8개(기술별 독립 호출 · 사실 단위 질의 · 정확도 임계값 없음 · 근거 없음 기록 · 출처 태그 강제 · 반대 근거 ≥2 · 중립성 검증 루프)를 적용했다. 판정 기준(재학습 불필요 · 전용 HW 불필요)은 배포 용이성에 가중을 두므로 구조적으로 SW 접근에 유리하다 — 도메인의 실제 제약을 반영한 것이지만 기준 선택 자체가 결과에 영향을 준다. Memory · Recall · Latency 3축은 판정이 아닌 해석에만 썼다." + viol_line,
        f"4. **`[추론]` 태그 비율** — 판단 문장 {st['tagged_total']}건 중 논문·웹 근거 없이 추론에만 의존한 문장 {ratio}.",
        f"5. **검색·생성 품질** — Hit Rate@4 {hit} · MRR@4 {mrr}{mode} · RAGAS Faithfulness {ragas.get('faithfulness', 'TBD')} · ResponseRelevancy {ragas.get('response_relevancy', 'TBD')} · ContextPrecision {ragas.get('context_precision', 'TBD')}. 검색 {st['retrieval_total']}회 중 \"논문에 근거 없음\" {no_ev}건({by_tech}). 임베딩 비교는 20문항 기준이라 0.10 차이는 2문항이며, 선정은 수치 우위가 아니라 한국어 질의 요건·컨텍스트 길이에 둔다. 근거 없음이 한 기술에 몰리면 그 기술 판정의 `[추론]` 비중이 높아진다. 웹 출처는 Tavily 가 저자·게시일을 주지 않는 경우가 많아 REFERENCE 에 기관명(사이트)과 접근일로 대체했다 — 게시일이 필요한 항목은 사람이 확인해야 한다.",
        f"6. **질의 재작성 효과** — {rewrite}. 효과가 없으면 재작성 단계 제거를 검토한다.",
    ])


# ---------- 6장 5번 — A 의 rag.evaluate 출력(outputs/eval.json) 매핑 ----------

ADOPTED_MODE = "dual-bm25/ko(dense)+en(sparse)"     # #27 실측에서 채택된 검색 구성 (README Tech Stack 과 같은 값)
_RAGAS_KEYS = {                                      # ragas 컬럼명 → 설계서 3.4 표기
    "faithfulness": "faithfulness",
    "answer_relevancy": "response_relevancy",
    "llm_context_precision_without_reference": "context_precision",
}


def metrics_from_eval(ev: dict[str, Any]) -> dict[str, Any]:
    """rag.evaluate 가 쓴 eval.json → render_limitations 의 retrieval_metrics.

    채택 구성(ADOPTED_MODE)이 있으면 그것을, 없으면 Hit@4 가 가장 높은 구성을 쓴다. 어느 구성인지 `mode` 로 남긴다.
    """
    retr = ev.get("retrieval") or {}
    mode = ADOPTED_MODE if ADOPTED_MODE in retr else (max(retr, key=lambda m: retr[m].get("hit@4", -1)) if retr else None)
    out: dict[str, Any] = {"mode": mode}
    if mode:
        out["hit@4"] = retr[mode].get("hit@4", "TBD")
        out["mrr@4"] = retr[mode].get("mrr@4", "TBD")
    ragas = ev.get("ragas") or {}
    out["ragas"] = {_RAGAS_KEYS.get(k, k): v for k, v in ragas.items()}
    if ev.get("rewrite"):
        out["rewrite"] = ev["rewrite"]           # {total, rewritten, rescued, still_no_evidence}
    return out


# ---------- 조립 ----------

def assemble(title: str, chapters: dict[str, str]) -> str:
    """CHAPTERS 순서로 조립. 없는 장은 '작성되지 않음' 으로 표시해 목차가 깨지지 않게 한다."""
    parts = [f"# {title}", ""]
    for key, heading in CHAPTERS:
        parts.append(f"## {heading}")
        parts.append("")
        parts.append(chapters.get(key) or "_(작성되지 않음)_")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"
