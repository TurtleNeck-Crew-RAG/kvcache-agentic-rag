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
