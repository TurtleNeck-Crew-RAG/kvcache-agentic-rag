"""보고서 렌더러 단위 테스트 — LLM 없이 도는 부분만 (설계서 6장). agents.report_render 는 langchain 을 import 하지 않는다."""
import json
from pathlib import Path

from agents.report_render import (
    CHAPTERS,
    assemble,
    format_ref,
    is_cited,
    limitation_stats,
    metrics_from_eval,
    render_evaluation,
    render_limitations,
    render_reference,
    render_selection,
)
from graph.state import init_state

FIX = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    d = json.loads((FIX / name).read_text(encoding="utf-8"))
    d.pop("_note", None)
    return d


def _full_state():
    s = init_state({}, {})
    s["tech_summary"] = _load("tech_summary.json")
    s.update(_load("evals.json"))
    s.update(_load("synthesis.json"))
    return s


def test_reference_excludes_uncited_and_dedups():
    cites = _load("synthesis.json")["citations"]
    body = "KIVI 는 2bit 양자화 [논문 p.1]. 구현은 https://github.com/jy-yuan/KIVI 참고 [웹 https://github.com/jy-yuan/KIVI]"
    ref = render_reference(cites + [cites[0]], body)       # 중복 1건 추가
    assert "미인용" not in ref                               # 본문에 없는 항목 제외
    assert ref.count("2402.02750") == 1                      # 중복 제거
    assert "2406.19707" in ref                               # [논문] 태그가 있으면 선정 논문 2편은 인용으로 본다
    assert "jy-yuan" in ref


def test_reference_format_matches_design_example():
    paper = _load("synthesis.json")["citations"][0]
    assert format_ref(paper) == (
        "Liu, Z., Yuan, J., Jin, H. et al.(2024). KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. "
        "*Proceedings of the 41st International Conference on Machine Learning (ICML), PMLR 235*. arXiv:2402.02750."
    )
    web = _load("synthesis.json")["citations"][2]
    assert format_ref(web).startswith("jy-yuan(") and format_ref(web).endswith("https://github.com/jy-yuan/KIVI")


def test_is_cited_web_requires_url_or_title():
    web = {"type": "웹", "authors": "x", "year": "2026", "title": "제목", "venue": "v", "id_or_url": "https://a.b/c", "accessed": ""}
    assert not is_cited(web, "[논문 p.1] 만 있는 본문")
    assert is_cited(web, "본문에 https://a.b/c 가 있다")


def test_evaluation_renders_both_techs_all_perspectives():
    md = render_evaluation(_full_state())
    for h in ("### 4.1", "### 4.2", "### 4.3", "### 4.4"):
        assert h in md
    assert md.count("**KIVI**") == 4 and md.count("**InfiniGen**") == 4
    assert "판정: 조건부" in md                               # DomainEval.verdict
    assert "권장" not in md and "더 낫다" not in md


def test_selection_renders_criteria_and_excluded():
    sel = {
        "sw": {"name": "KIVI", "paper": "P", "venue": "ICML 2024", "arxiv": "2402.02750", "pages": 15, "reason": "r"},
        "hw": {"name": "InfiniGen", "paper": "Q", "venue": "OSDI 2024", "arxiv": "2406.19707", "pages": 18, "reason": "r"},
        "criteria": [{"key": "k", "label": "L", "question": "Q?"}],
        "excluded": [{"name": "DeepSeek-V2 MLA", "side": "sw", "reason": "재학습"}],
    }
    md = render_selection(sel)
    assert "**L** — Q?" in md and "DeepSeek-V2 MLA (SW) — 재학습" in md and "15p" in md


def test_limitation_stats_and_rendering():
    s = _full_state()
    s["retrieval_log"] = [
        {"node": "tech", "tech": "KIVI", "query_before": "q", "query_after": None, "hits_before": ["c1"], "hits_after": None, "relevance": "yes", "rewritten": False},
        {"node": "tech", "tech": "InfiniGen", "query_before": "q", "query_after": "q2", "hits_before": [], "hits_after": ["c9"], "relevance": "yes", "rewritten": True},
        {"node": "domain", "tech": "InfiniGen", "query_before": "q", "query_after": "q2", "hits_before": [], "hits_after": [], "relevance": "no_evidence", "rewritten": True},
    ]
    st = limitation_stats(s)
    assert st["tagged_total"] > 0 and 0 <= st["inference_ratio"] <= 1
    assert st["retrieval_total"] == 3 and st["no_evidence"] == 1 and st["by_tech_no_evidence"]["InfiniGen"] == 1
    assert st["rewritten"] == 2 and st["rewrite_recovered"] == 1
    md = render_limitations(s, st, {"hit@4": 0.8, "mrr@4": 0.6, "ragas": {"faithfulness": 0.9}})
    assert md.startswith("1. ") and "6. " in md
    assert "Hit Rate@4 0.8" in md and "재작성 2건 중 1건" in md and "InfiniGen 1건" in md
    assert "ResponseRelevancy TBD" in md                      # 없는 수치는 TBD


def test_assemble_order_summary_first_reference_last():
    md = assemble("T", {"summary": "S", "reference": "R"})
    heads = [line for line in md.splitlines() if line.startswith("## ")]
    assert heads[0] == "## SUMMARY" and heads[-1] == "## REFERENCE"
    assert len(heads) == len(CHAPTERS)
    assert "_(작성되지 않음)_" in md                           # 빠진 장은 표시만, 목차는 유지


def test_metrics_from_eval_maps_adopted_mode_and_ragas_keys():
    ev = {
        "retrieval": {
            "dense/ko": {"hit@4": 0.55, "mrr@4": 0.4, "miss": [4]},
            "dual-bm25/ko(dense)+en(sparse)": {"hit@4": 0.8, "mrr@4": 0.52, "miss": [3, 8]},
        },
        "rewrite": {"total": 10, "rewritten": 3, "rescued": 3, "still_no_evidence": 0},
        "ragas": {"faithfulness": 0.936, "answer_relevancy": 0.838, "llm_context_precision_without_reference": 0.912},
    }
    m = metrics_from_eval(ev)
    assert m["mode"] == "dual-bm25/ko(dense)+en(sparse)" and m["hit@4"] == 0.8 and m["mrr@4"] == 0.52
    assert m["ragas"] == {"faithfulness": 0.936, "response_relevancy": 0.838, "context_precision": 0.912}
    md = render_limitations(_full_state(), None, m)
    assert "Hit Rate@4 0.8" in md and "ResponseRelevancy 0.838" in md and "ContextPrecision 0.912" in md
    assert "dual-bm25/ko(dense)+en(sparse)" in md


def test_metrics_from_eval_falls_back_to_best_mode():
    m = metrics_from_eval({"retrieval": {"a": {"hit@4": 0.3, "mrr@4": 0.2}, "b": {"hit@4": 0.6, "mrr@4": 0.5}}})
    assert m["mode"] == "b" and m["hit@4"] == 0.6 and m["ragas"] == {}
