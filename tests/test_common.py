from agents._common import PROMPT_DIR, render_prompt


def test_render_prompt_keeps_other_braces(tmp_path, monkeypatch):
    (tmp_path / "x.md").write_text("Q: {question}\n출력 {relevant: yes|no}", encoding="utf-8")
    monkeypatch.setattr("agents._common.PROMPT_DIR", tmp_path)
    assert render_prompt("x", question="왜?") == "Q: 왜?\n출력 {relevant: yes|no}"


def test_all_rag_prompts_exist():
    for name in ("rag_relevance", "rag_rewrite", "rag_generator", "rag_faithfulness"):
        assert (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def test_every_llm_role_has_call_timeout_and_retries(monkeypatch):
    """#102 — 호출 1회 상한. 없으면 OpenAI 기본 600초라 노드 안에서 멈춘 호출을 RUN_TIMEOUT 도 못 끊는다."""
    from agents import _common

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")                 # 객체 생성만 — 호출하지 않는다
    for role in _common.MODELS:
        m = _common.llm(role)
        assert m.request_timeout == 60 and m.max_retries == 2, role
