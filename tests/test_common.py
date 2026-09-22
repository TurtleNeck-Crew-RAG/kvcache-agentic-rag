from agents._common import PROMPT_DIR, render_prompt


def test_render_prompt_keeps_other_braces(tmp_path, monkeypatch):
    (tmp_path / "x.md").write_text("Q: {question}\n출력 {relevant: yes|no}", encoding="utf-8")
    monkeypatch.setattr("agents._common.PROMPT_DIR", tmp_path)
    assert render_prompt("x", question="왜?") == "Q: 왜?\n출력 {relevant: yes|no}"


def test_all_rag_prompts_exist():
    for name in ("rag_relevance", "rag_rewrite", "rag_generator", "rag_faithfulness"):
        assert (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
