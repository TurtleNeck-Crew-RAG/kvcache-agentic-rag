"""run_summary — app.py 출력 · 결정 로그로 실행 요약. 파일은 임시 폴더."""
import json

import pytest

import graph.observe as ob
import graph.run_summary as rs


@pytest.fixture(autouse=True)
def _out(monkeypatch, tmp_path):
    monkeypatch.setattr(ob, "OUT", tmp_path)
    return tmp_path


def _write(tmp, name, data):
    (tmp / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _decide(step, decision, reason, trace="T"):
    ob.log_decision({"trace_id": trace, "step_count": step}, "supervisor", decision, reason)


def test_summary_counts_routing_reworks_and_unresolved(_out):
    _write(_out, "run.json", {"trace_id": "T", "status": "SUCCESS", "step_count": 6, "llm_calls": 120,
                              "eval_attempts": 1, "retry": {"market:InfiniGen": 2}, "elapsed_sec": 300})
    _write(_out, "sufficiency.json", {
        "market:InfiniGen": {"rule": "fail", "judge": None, "gap": "source_bias", "reason": "github 88% — 과반"},
        "market:KIVI": {"rule": "pass", "judge": "sufficient", "gap": "", "reason": "ok"}})
    _write(_out, "eval_result.json", {"passed": True, "items": {"bias": {"passed": True, "score": 0.8, "reason": "r"}}})
    _decide(0, "market", "미수집 셀 market:KIVI, market:InfiniGen")
    _decide(1, "market", "부족 셀 market:InfiniGen 재작업 1/2 — github 88%")
    _decide(2, "market", "부족 셀 market:InfiniGen 재작업 2/2 — github 80%")
    _decide(3, "synthesis", "수집 셀 전부 충분 또는 재작업 소진")
    _decide(4, "report", "평가 fail(groundedness) 루프 1/2 → 보고서 재작성")
    _decide(5, "__end__", "보고서 평가 pass")
    _decide(0, "tech_research", "다른 실행", trace="OTHER")

    s = rs.summarize()
    assert s["trace_id"] == "T" and s["routing_count"] == 6
    assert s["reworks"] == {"market:InfiniGen": 2} and s["rework_total"] == 2 and s["eval_loops"] == 1
    assert list(s["unresolved"]) == ["market:InfiniGen"]
    md = rs.to_markdown(s)
    assert "셀 재작업 | 2회 — market:InfiniGen ×2" in md and "끝까지 부족한 셀" in md and "다른 실행" not in md


def test_other_trace_shows_decisions_only(_out):
    _write(_out, "run.json", {"trace_id": "LAST"})
    _decide(0, "tech_research", "미수집 셀", trace="OLD")
    s = rs.summarize("OLD")
    assert s["run"] == {} and s["sufficiency"] == {} and s["routing_count"] == 1
    assert "다른 실행 것" in rs.to_markdown(s)


def test_main_writes_markdown(_out):
    _write(_out, "run.json", {"trace_id": "T"})
    _decide(0, "__end__", "보고서 평가 pass")
    assert rs.main(["--md"]) == 0
    assert (_out / "run_summary.md").read_text(encoding="utf-8").startswith("# 실행 요약 — `T`")


def test_rework_effect_tracks_verdict_per_round(_out):
    _write(_out, "run.json", {"trace_id": "T", "retry": {}})
    _write(_out, "sufficiency.json", {
        "market:InfiniGen": {"rule": "fail", "judge": None, "reason": "출처 편중 github.com 70% — 과반"},
        "domain:KIVI": {"rule": "pass", "judge": "sufficient", "reason": "ok"},
        "stakeholder:KIVI": {"rule": "fail", "judge": None, "reason": "반대 근거 1건 < 2"}})
    _decide(1, "market", "부족 셀 market:InfiniGen 재작업 1/2 — 출처 편중 github.com 88% — 과반")
    _decide(2, "market", "부족 셀 market:InfiniGen 재작업 2/2 — 출처 편중 github.com 75% — 과반")
    _decide(3, "domain", "부족 셀 domain:KIVI 재작업 1/2 — 웹 근거 0건 < 1 (웹 반례 필요)")
    _decide(4, "stakeholder", "부족 셀 stakeholder:KIVI 재작업 1/2 — 반대 근거 1건 < 2")
    _decide(5, "stakeholder", "부족 셀 stakeholder:KIVI 재작업 2/2 — 반대 근거 1건 < 2")
    rows = {r["cell"]: r for r in rs.summarize()["rework_effect"]}
    assert rows["domain:KIVI"]["effect"] == "해소" and rows["domain:KIVI"]["states"][-1] == "충분"
    assert rows["market:InfiniGen"]["effect"] == "개선"                 # 88 → 75 → 70
    assert rows["stakeholder:KIVI"]["effect"] == "변화 없음"            # 2회째가 판정을 못 바꿈
    md = rs.to_markdown(rs.summarize())
    assert "재작업 회차별 판정 변화" in md and "2회째 재작업이 판정을 바꾼 셀: **1/2**" in md


def test_judge_wording_change_is_not_improvement(_out):
    # 같은 문제(Judge off_topic)로 다시 부족한데 LLM 문장만 바뀐 경우 — "개선" 이 아니라 "변화 없음"
    _write(_out, "run.json", {"trace_id": "T"})
    _write(_out, "sufficiency.json", {"market:KIVI": {
        "rule": "pass", "judge": "insufficient", "gap": "off_topic", "reason": "근거 6건 · Judge: 시장 근거가 여전히 기술 수치 위주"}})
    _decide(1, "market", "부족 셀 market:KIVI 재작업 1/2 — gap=off_topic · 근거 6건 · Judge: 시장 규모 근거 부족")
    _decide(2, "market", "부족 셀 market:KIVI 재작업 2/2 — gap=off_topic · 근거 7건 · Judge: 수요 근거가 없음")
    row = rs.summarize()["rework_effect"][0]
    assert row["effect"] == "변화 없음"
    assert rs._verdict_key("근거 6건 · Judge: 표현 A") == rs._verdict_key("근거 6건 · Judge: 표현 B")


def test_web_calls_counted_per_node(_out):
    _write(_out, "run.json", {"trace_id": "T", "llm_calls": 1})
    for node, n in (("market", 5), ("market", 0), ("domain", 2), ("stakeholder", 4)):
        ob.log_web("T", node, "q", n)
    ob.log_web("OTHER", "market", "q", 1)
    s = rs.summarize()
    assert s["web_calls"] == {"market": 2, "domain": 1, "stakeholder": 1} and s["web_empty"] == 1
    assert "웹 검색 (Tavily) | 4회 — domain 1, market 2, stakeholder 1 (결과 0건 1회)" in rs.to_markdown(s)
