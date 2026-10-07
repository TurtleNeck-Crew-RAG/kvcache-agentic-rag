"""보고서 워커의 evaluator feedback · report_uri · 10쪽 압축 계약. [소유: B 심준용]"""
from pathlib import Path
from types import SimpleNamespace

from agents import report


def test_write_chapter_includes_evaluator_feedback(monkeypatch):
    seen = {}

    def invoke(messages):
        seen["messages"] = messages
        return SimpleNamespace(content="수정된 장")

    monkeypatch.setattr(report, "llm", lambda role: SimpleNamespace(invoke=invoke))
    result = report._write_chapter("공통", "장 지시", {"domain": {}}, "REFERENCE 대응을 보강")
    assert result == "수정된 장"
    assert "이전 보고서 평가 피드백" in seen["messages"][1][1]
    assert "REFERENCE 대응을 보강" in seen["messages"][1][1]


def test_build_report_passes_feedback_to_all_llm_chapters(monkeypatch):
    prompt = """# 공통 규칙
공통
# chapter: background
배경
# chapter: overview
개요
# chapter: implications
시사점
# chapter: summary
요약
"""
    seen = []

    def write(common, instruction, payload, feedback=""):
        seen.append((instruction, feedback))
        return instruction

    monkeypatch.setattr(report, "load_prompt", lambda name: prompt)
    monkeypatch.setattr(report, "_write_chapter", write)
    monkeypatch.setattr(report, "_load_metrics", lambda: None)
    state = {
        "domain": {}, "selected": {}, "tech_summary": {}, "citations": [],
        "synthesis": {"matrix": {}, "agreements": [], "conflicts": []},
        "trl_estimate": {},
        "eval_result": {"feedback": "중립성 표현을 수정"},
    }
    report_md, calls = report.build_report(state)
    assert calls == 4 and len(seen) == 4
    assert all(feedback == "중립성 표현을 수정" for _, feedback in seen)
    assert "## SUMMARY" in report_md and "## REFERENCE" in report_md


def test_run_compacts_only_evaluation_when_pdf_exceeds_ten_pages(monkeypatch, tmp_path):
    full = "# 보고서\n\n## 4. 관점별 평가\n\nFULL\n\n## 5. 시사점\n\nKEEP"
    saved = []

    monkeypatch.setattr(report, "build_report", lambda state: (full, 4))
    monkeypatch.setattr(report, "render_evaluation", lambda state, compact=False: "COMPACT" if compact else "FULL")

    def save_with_pages(text):
        saved.append(text)
        return tmp_path / "report.md", 11 if len(saved) == 1 else 9

    monkeypatch.setattr(report, "save_with_pages", save_with_pages)
    result = report.run({})
    assert result == {"report_uri": (tmp_path / "report.md").as_posix(), "llm_calls": 4}
    assert len(saved) == 2
    assert "FULL" in saved[0] and "COMPACT" in saved[1]
    assert "## 5. 시사점\n\nKEEP" in saved[1]


def test_run_returns_report_uri_when_pdf_renderer_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "build_report", lambda state: ("# 보고서", 4))
    monkeypatch.setattr(report, "save_with_pages", lambda text: (tmp_path / "report.md", None))
    assert report.run({}) == {
        "report_uri": (tmp_path / "report.md").as_posix(),
        "llm_calls": 4,
    }


def test_save_with_pages_writes_markdown_and_preserves_relative_uri(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "_to_pdf_with_pages", lambda path: (None, None))
    md, pages = report.save_with_pages("# 결과", tmp_path)
    assert md == Path(tmp_path) / "report.md" and pages is None
    assert md.read_text(encoding="utf-8") == "# 결과"
