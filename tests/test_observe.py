"""observe — 파일 적재 · trace_id 상관. LangSmith 없이 돈다."""
import graph.observe as ob


def test_trace_id_is_unique_and_readable():
    a, b = ob.new_trace_id(), ob.new_trace_id()
    assert a != b and a[:8].isdigit()


def test_run_config_ties_thread_id_and_metadata():
    cfg = ob.run_config("t-1", 100)
    assert cfg["configurable"]["thread_id"] == cfg["metadata"]["trace_id"] == "t-1"
    assert cfg["recursion_limit"] == 100


def test_decisions_and_retrieval_go_to_jsonl_by_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "OUT", tmp_path)
    ob.log_decision({"trace_id": "A", "step_count": 3}, "supervisor", "market", "미수집 셀")
    ob.log_decision({"trace_id": "B"}, "supervisor", "__end__", "평가 pass")
    ob.log_retrieval("A", [{"node": "tech_research", "tech": "KIVI", "relevance": "yes"}])

    a = ob.read_jsonl(ob.DECISIONS, "A")
    assert len(a) == 1 and a[0]["step"] == 3 and a[0]["reason"] == "미수집 셀"
    assert len(ob.read_jsonl(ob.DECISIONS)) == 2
    r = ob.read_jsonl(ob.RETRIEVAL, "A")
    assert r[0]["tech"] == "KIVI" and r[0]["trace_id"] == "A"


def test_log_retrieval_empty_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "OUT", tmp_path)
    ob.log_retrieval("A", [])
    assert ob.read_jsonl(ob.RETRIEVAL) == []


def test_retrieval_entries_prefers_file_then_state(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "OUT", tmp_path)
    st = {"trace_id": "A", "retrieval_log": [{"tech": "KIVI", "relevance": "no"}]}
    assert ob.retrieval_entries(st) == st["retrieval_log"]                 # 파일 없음 → State
    ob.log_retrieval("A", [{"tech": "InfiniGen", "relevance": "yes"}])
    ob.log_retrieval("B", [{"tech": "KIVI", "relevance": "yes"}])
    rows = ob.retrieval_entries(st)
    assert [r["tech"] for r in rows] == ["InfiniGen"]                      # 이 실행 것만, 파일이 정본


def test_log_web_by_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "OUT", tmp_path)
    ob.log_web("A", "market", "KIVI 시장 채택", 5)
    ob.log_web("A", "stakeholder", "KIVI 비판", 0)            # 실패 질의도 남긴다 — 크레딧은 쓰였다
    ob.log_web("B", "domain", "InfiniGen 반례", 3)
    rows = ob.read_jsonl(ob.WEB, "A")
    assert [(r["node"], r["n_results"]) for r in rows] == [("market", 5), ("stakeholder", 0)]
