"""app.py 의 부분 State 저장 — 그래프가 중간에 죽어도 outputs/run.json 과 채워진 키의 JSON 이 남는지. LLM 불필요."""
import json
import sys
import types

import pytest


@pytest.fixture
def app_module(monkeypatch, tmp_path):
    # app.py 는 graph.build 를 import 하고, graph.build 는 agents.* 를 import 한다.
    # agents._common 이 langchain_openai 를 끌어오므로 CI(langgraph + pytest 만)에서는 스텁으로 막는다.
    stub = types.ModuleType("agents._common")
    stub.llm = lambda role: None
    stub.load_prompt = lambda name: ""
    monkeypatch.setitem(sys.modules, "agents._common", stub)
    for mod, attr in (("yaml", "safe_load"), ("dotenv", "load_dotenv")):
        if mod not in sys.modules:
            m = types.ModuleType(mod)
            setattr(m, attr, lambda *a, **k: {})
            monkeypatch.setitem(sys.modules, mod, m)
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
