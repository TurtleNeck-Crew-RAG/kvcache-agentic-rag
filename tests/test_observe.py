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
