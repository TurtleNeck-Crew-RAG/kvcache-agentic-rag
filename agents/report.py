"""보고서 생성 에이전트 — 설계서 6장 목차 (SUMMARY ½p → 1~6장 → REFERENCE).

출력 키: report_md · llm_calls (+ outputs/report/report.md → PDF)

LLM 이 쓰는 장: 1 배경 · 3 개요 · 5 시사점 · SUMMARY(맨 마지막 — 본문 전체를 입력으로). 4회 호출.
State 만으로 만드는 장: 2 선정 · 4 관점별 평가 · 6 한계점 · REFERENCE → agents/report_render.py
장별로 호출을 나누는 이유: 컨텍스트가 터지지 않고, 장마다 분량·규칙을 따로 걸 수 있다 (ROLES.md D).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from agents._common import llm, load_prompt
from agents.report_render import (
    arxiv_id_from_url,
    assemble,
    is_failed,
    limitation_stats,
    metrics_from_eval,
    normalize_citations,
    render_evaluation,
    render_limitations,
    render_reference,
    render_selection,
)
from graph.state import GraphState

TITLE = "KV cache 최적화 기술 다관점 평가 — KIVI(SW) · InfiniGen(HW) · 스마트폰 온디바이스 LLM"
OUT_DIR = Path("outputs/report")
EVAL_PATHS = (Path("outputs/eval.json"),                  # A 의 rag.evaluate 출력 (#27) → 6장 5번
              Path("experiments/sparse_compare/eval.json"))  # 리포에 커밋된 실측 사본 (outputs/ 는 git 제외)
METRICS_PATH = Path("outputs/retrieval_metrics.json")     # 수동으로 넣을 때의 대체 경로 {"hit@4","mrr@4","ragas"}
ARXIV_META_CACHE = Path("data/cache/arxiv_meta.json")    # arXiv API 응답 캐시 (git 제외) — 웹검색이 긁어 온 논문 페이지의 저자·게시일


def _split_prompt(text: str) -> tuple[str, dict[str, str]]:
    """prompts/report.md → (공통 규칙, {chapter_key: 지시})."""
    common, chapters, cur = [], {}, None
    for line in text.splitlines():
        if line.startswith("# chapter: "):
            cur = line.split(":", 1)[1].strip()
            chapters[cur] = []
        elif line.startswith("# 공통 규칙"):
            cur = None
        elif cur is None:
            common.append(line)
        else:
            chapters[cur].append(line)
    return "\n".join(common).strip(), {k: "\n".join(v).strip() for k, v in chapters.items()}


def _load_metrics() -> dict[str, Any] | None:
    for p in EVAL_PATHS:
        if p.exists():
            return metrics_from_eval(json.loads(p.read_text(encoding="utf-8")))
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return None


def _arxiv_meta(ids: list[str]) -> dict[str, dict[str, Any]]:
    """export.arxiv.org API 로 저자·제목·게시일을 받는다. 캐시 우선, 실패하면 빈 dict (REFERENCE 는 제목·id 만으로 렌더링)."""
    cache: dict[str, dict[str, Any]] = {}
    if ARXIV_META_CACHE.exists():
        try:
            cache = json.loads(ARXIV_META_CACHE.read_text(encoding="utf-8"))
        except Exception:                          # noqa: BLE001 — 캐시 손상은 무시하고 다시 받는다
            cache = {}
    missing = [i for i in dict.fromkeys(ids) if i not in cache]
    if missing:
        import urllib.request
        import xml.etree.ElementTree as ET
        ns = {"a": "http://www.w3.org/2005/Atom"}
        url = "https://export.arxiv.org/api/query?id_list=" + ",".join(missing) + f"&max_results={len(missing)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kvcache-agentic-rag/0.1 (SKALA course project)"})
            with urllib.request.urlopen(req, timeout=10) as r:            # noqa: S310 — 고정 도메인
                root = ET.fromstring(r.read())
            for e in root.findall("a:entry", ns):
                eid = (e.findtext("a:id", default="", namespaces=ns) or "").rsplit("/", 1)[-1]
                eid = re.sub(r"v\d+$", "", eid)
                cache[eid] = {
                    "title": " ".join((e.findtext("a:title", default="", namespaces=ns) or "").split()),
                    "authors": [a.findtext("a:name", default="", namespaces=ns) for a in e.findall("a:author", ns)],
                    "published": e.findtext("a:published", default="", namespaces=ns) or "",
                }
            ARXIV_META_CACHE.parent.mkdir(parents=True, exist_ok=True)
            ARXIV_META_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as e:                     # noqa: BLE001 — 네트워크 실패는 보고서를 막지 않는다
            print(f"[report] arXiv 메타 조회 실패({type(e).__name__}) — 제목·id 로만 표기", file=sys.stderr)
    return {i: cache[i] for i in ids if i in cache}


def _pick(state: GraphState, *keys: str) -> dict[str, Any]:
    return {k: state.get(k) for k in keys if state.get(k) is not None}


def _write_chapter(common: str, instruction: str, payload: dict[str, Any]) -> str:
    msgs = [
        ("system", common),
        ("human", instruction + "\n\n## 입력 (JSON)\n```json\n" + json.dumps(payload, ensure_ascii=False, indent=1) + "\n```"),
    ]
    return llm("generator").invoke(msgs).content.strip()


def build_report(state: GraphState) -> tuple[str, int]:
    """report_md 와 LLM 호출 수를 돌려준다. 파일 저장은 하지 않는다."""
    common, instr = _split_prompt(load_prompt("report"))
    calls = 0
    ch: dict[str, str] = {}

    # State 만으로 만드는 장
    ch["selection"] = render_selection(state.get("selected") or {})
    ch["evaluation"] = render_evaluation(state)
    metrics = _load_metrics()
    ch["limitations"] = render_limitations(state, limitation_stats(state), metrics)

    # LLM 이 쓰는 장 — 배경 · 개요 · 시사점
    ch["background"] = _write_chapter(common, instr["background"], _pick(state, "domain"))
    calls += 1
    ch["overview"] = _write_chapter(common, instr["overview"], _pick(state, "tech_summary"))
    calls += 1
    if is_failed(state.get("synthesis")):     # 종합이 실패 기록이면 LLM 이 "실패" 를 엇갈림 주제로 쓴다 (실측 3회차) → 고정 문장
        reason = next((c for c in (state["synthesis"].get("conflicts") or []) if "실패" in c), "종합 워커 실패")
        ch["implications"] = (
            "종합 워커가 실행되지 않아 관점 × 기술 매트릭스와 엇갈림 분석을 생성하지 못했다. "
            f"사유: {reason}. 관점별 원자료는 4장에 그대로 있다. 시사점은 종합 워커 복구 후 다시 생성해야 한다."
        )
    else:
        ch["implications"] = _write_chapter(common, instr["implications"], _pick(state, "synthesis", "trl_estimate"))
        calls += 1

    # REFERENCE — 본문(1~6장)에 인용된 것만
    body = "\n".join(ch[k] for k in ("background", "selection", "overview", "evaluation", "implications", "limitations"))
    raw = state.get("citations") or []
    ids = [a for a in (arxiv_id_from_url(str(c.get("id_or_url") or "")) for c in raw if c.get("type") == "웹") if a]
    cites = normalize_citations(raw, state.get("selected") or {}, _arxiv_meta(ids) if ids else {})
    corpus_ids = {str(p.get("arxiv")) for k in ("sw", "hw") for p in [(state.get("selected") or {}).get(k) or {}] if p.get("arxiv")}
    ch["reference"] = render_reference(cites, body, corpus_ids)

    # SUMMARY — 맨 마지막, 본문 전체를 입력으로
    ch["summary"] = _write_chapter(common, instr["summary"], {"report_body": assemble(TITLE, {k: v for k, v in ch.items() if k != "summary"})})
    calls += 1

    return assemble(TITLE, ch), calls


def save(report_md: str, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "report.md"
    md.write_text(report_md, encoding="utf-8")
    pdf = to_pdf(md)
    if pdf:
        print(f"report: {md} → {pdf}")
    else:
        print(f"report: {md} (PDF 는 weasyprint 미설치 — VS Code Markdown PDF 로 수동 변환)")
    return md


def to_pdf(md_path: Path) -> Path | None:
    """markdown → HTML → PDF (weasyprint). 시스템 pango 가 없으면 None 을 돌려주고 md 만 남긴다."""
    try:
        import markdown
        from weasyprint import HTML
    except Exception:
        return None
    html = markdown.markdown(md_path.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
    css = "body{font-family:'Apple SD Gothic Neo','Noto Sans KR',sans-serif;font-size:10.5pt;line-height:1.5;margin:2cm} h1{font-size:18pt} h2{font-size:14pt;margin-top:1.4em;border-bottom:1px solid #999} code{font-size:9pt}"
    pdf_name = os.environ.get("REPORT_PDF_NAME", "report.pdf")
    pdf = md_path.with_name(pdf_name)
    HTML(string=f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{html}</body></html>").write_pdf(pdf)
    return pdf


def run(state: GraphState) -> dict:
    report_md, calls = build_report(state)
    save(report_md)
    return {"report_md": report_md, "llm_calls": calls}


if __name__ == "__main__":
    # fixtures 로 단독 실행: uv run python -m agents.report  (LLM 4회 호출 — OPENAI_API_KEY 필요)
    import yaml
    from dotenv import load_dotenv

    from graph.state import init_state

    load_dotenv()
    fx = Path("tests/fixtures")

    def _load(name: str) -> dict:
        d = json.loads((fx / name).read_text(encoding="utf-8"))
        d.pop("_note", None)
        return d

    s = init_state(yaml.safe_load(Path("config/domain.yaml").read_text()), yaml.safe_load(Path("config/selection.yaml").read_text()))
    s["tech_summary"] = _load("tech_summary.json")
    s.update(_load("evals.json"))
    s.update(_load("synthesis.json"))
    out = run(s)
    print(f"llm_calls={out['llm_calls']}  chars={len(out['report_md'])}")
