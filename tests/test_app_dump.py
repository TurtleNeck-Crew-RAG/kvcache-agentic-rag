"""app.py 의 부분 State 저장 — 그래프가 중간에 죽어도 outputs/run.json 과 채워진 키의 JSON 이 남는지. LLM 불필요."""
import json
import sys
import types

import pytest


@pytest.fixture
def app_module(monkeypatch, tmp_path):
    # app.py 는 graph.build 를 import 하고, graph.build 는 agents.* → rag.* (chromadb · FlagEmbedding …) 를 끌어온다.
    # 이 테스트는 app._dump 만 보므로 graph.build 자체를 스텁으로 막는다 — 워커가 무엇을 import 하든 영향 없음.
    stub = types.ModuleType("graph.build")
    stub.build_graph = lambda **kw: None
    monkeypatch.setitem(sys.modules, "graph.build", stub)
    for mod, attr in (("yaml", "safe_load"), ("dotenv", "load_dotenv")):
        try:
            __import__(mod)
        except ImportError:                       # CI(langgraph + pytest 만) 에서만 더미
            m = types.ModuleType(mod)
            setattr(m, attr, lambda *a, **k: {})
            monkeypatch.setitem(sys.modules, mod, m)
    monkeypatch.delitem(sys.modules, "app", raising=False)
    import app
    monkeypatch.setattr(app, "OUT", tmp_path)
    return app


def test_dump_writes_run_json_and_filled_keys_only(app_module, tmp_path):
    state = {"llm_calls": 7, "retry": {"market:InfiniGen": 1}, "citations": [{"title": "x"}], "synthesis": None,
             "market_eval": {}, "trace_id": "t-1", "step_count": 3, "node_status": {"market": "failed"}}
    app_module._dump(state, ["start", "tech_research", "market"], "RuntimeError: boom", 3.14)
    run = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert run["ok"] is False and "boom" in run["error"]
    assert run["visited"] == ["start", "tech_research", "market"]
    assert run["llm_calls"] == 7 and run["retry"] == {"market:InfiniGen": 1}
    assert run["trace_id"] == "t-1" and run["step_count"] == 3 and run["node_status"] == {"market": "failed"}
    assert run["keys_filled"] == ["citations"]
    assert (tmp_path / "citations.json").exists()
    assert not (tmp_path / "synthesis.json").exists()          # None 인 키는 파일을 만들지 않는다
    assert not (tmp_path / "market_eval.json").exists()        # init_state 의 빈 {} 도 만들지 않는다 (실측 2026-09-22 — 2바이트 파일 5개)


def test_pdf_name_is_exported_before_graph_runs(app_module, monkeypatch):
    """--pdf-name 은 보고서 워커가 읽는 env 로 그래프 실행 전에 넘어가야 한다 (4회차 실측: 안 넘어가 report.pdf 로만 저장됨)."""
    import os

    seen = {}

    def fake_build_graph(**kw):
        seen["env"] = os.environ.get("REPORT_PDF_NAME")
        raise RuntimeError("stop")               # 그래프는 실행하지 않는다

    monkeypatch.setattr(app_module, "build_graph", fake_build_graph)
    monkeypatch.setattr(app_module, "_ensure_index", lambda skip: None)
    monkeypatch.setattr(app_module.yaml, "safe_load", lambda *_: {})
    monkeypatch.setattr(app_module.Path, "read_text", lambda *_a, **_k: "")
    monkeypatch.delenv("REPORT_PDF_NAME", raising=False)
    try:
        app_module.main(["--skip-index", "--pdf-name", "RAG-Output_test.pdf"])
    except RuntimeError:
        pass
    assert seen["env"] == "RAG-Output_test.pdf"


def test_main_runs_stub_graph_to_end_with_trace_id(app_module, monkeypatch, tmp_path):
    """app.main — stub 그래프로 trace_id 생성 → 체크포인터 → stream → run.json 까지 (LLM 없음)."""
    from graph.build import (
        build_graph as real_build,  # noqa: F401 — 스텁 모듈이 아니라 진짜를 다시 가져온다
    )
    monkeypatch.delitem(sys.modules, "graph.build", raising=False)
    import importlib

    gb = importlib.import_module("graph.build")                # 스텁이 아니라 진짜 graph.build
    from tests.fixtures.stubs import make_assess, make_evaluator, make_workers

    workers = make_workers()                                     # fixtures 를 읽은 뒤에 Path.read_text 를 막는다
    monkeypatch.setattr(app_module, "build_graph", lambda **kw: gb.build_graph(
        workers=workers, assess=make_assess(), evaluator=make_evaluator(fail_times=1), **kw))
    monkeypatch.setattr(app_module, "_ensure_index", lambda skip: None)
    monkeypatch.setattr(app_module, "_save_report", lambda state, visited: None)
    monkeypatch.setattr(app_module.yaml, "safe_load", lambda *_: {})
    monkeypatch.setattr(app_module.Path, "read_text", lambda *_a, **_k: "")
    monkeypatch.setattr(app_module, "make_checkpointer", lambda: (__import__("langgraph.checkpoint.memory", fromlist=["x"]).InMemorySaver(), False))
    assert app_module.main(["--skip-index"]) == 0
    with open(tmp_path / "run.json", encoding="utf-8") as f:     # Path.read_text 는 위에서 막혀 있다
        run = json.load(f)
    assert run["status"] == "SUCCESS" and run["trace_id"] and run["eval_attempts"] == 1
    assert run["visited"].count("report") == 2 and run["visited"][-1] == "evaluator"
    assert app_module.main(["--resume", run["trace_id"]]) == 2     # InMemorySaver 로는 프로세스를 넘는 재개 불가 — 명시적으로 거절


def test_main_timeout_stops_at_node_boundary_as_interrupted(app_module, monkeypatch, tmp_path):
    """#102 — 벽시계 상한은 게이트 밖(app.py). 넘으면 노드 경계에서 멈추고 INTERRUPTED · tokens 기록."""
    import importlib

    from langgraph.checkpoint.memory import InMemorySaver

    from tests.fixtures.stubs import make_assess, make_evaluator, make_workers

    monkeypatch.delitem(sys.modules, "graph.build", raising=False)
    gb = importlib.import_module("graph.build")
    workers = make_workers()
    monkeypatch.setattr(app_module, "build_graph", lambda **kw: gb.build_graph(
        workers=workers, assess=make_assess(), evaluator=make_evaluator(), **kw))
    monkeypatch.setattr(app_module, "_ensure_index", lambda skip: None)
    monkeypatch.setattr(app_module, "_save_report", lambda state, visited: None)
    monkeypatch.setattr(app_module.yaml, "safe_load", lambda *_: {})
    monkeypatch.setattr(app_module.Path, "read_text", lambda *_a, **_k: "")
    monkeypatch.setattr(app_module, "make_checkpointer", lambda: (InMemorySaver(), False))
    assert app_module.main(["--skip-index", "--timeout", "0"]) == 1
    with open(tmp_path / "run.json", encoding="utf-8") as f:
        run = json.load(f)
    assert run["status"] == "INTERRUPTED" and run["error"].startswith("Timeout")
    assert run["visited"] == [] and run["step_count"] == 1 and "tokens" in run   # 첫 노드(supervisor) 경계에서 멈춤
