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
    stub.build_graph = lambda: None
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
    state = {"llm_calls": 7, "retry": {"stake": 1}, "citations": [{"title": "x"}], "synthesis": None}
    app_module._dump(state, ["start", "tech_research", "market"], "RuntimeError: boom", 3.14)
    run = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert run["ok"] is False and "boom" in run["error"]
    assert run["visited"] == ["start", "tech_research", "market"]
    assert run["llm_calls"] == 7 and run["retry"] == {"stake": 1}
    assert run["keys_filled"] == ["citations"]
    assert (tmp_path / "citations.json").exists()
    assert not (tmp_path / "synthesis.json").exists()          # None 인 키는 파일을 만들지 않는다
