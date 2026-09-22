"""기술 조사 워커 — ask() 를 monkeypatch 해서 조립 로직만 검증."""
import pytest

pytest.importorskip("langchain_chroma")
pytest.importorskip("FlagEmbedding")

import agents.tech_research as tr  # noqa: E402
from rag.rag_node import NO_EVIDENCE  # noqa: E402


def test_questions_have_five_keys_and_fallbacks():
    qs = dict(tr.questions("KIVI"))
    assert list(qs) == ["overview", "mechanism", "numbers", "limitations", "apply_conditions"]
    assert all("KIVI" in v for vs in qs.values() for v in vs)
    assert len(qs["limitations"]) == 2          # 대체 질문


def test_sentences_split_only_after_page_tag():
    assert tr._sentences("A 감소. 정확도 유지[p.2][p.1]. 배치 4×[p.1].") == ["A 감소. 정확도 유지[p.2][p.1]", "배치 4×[p.1]"]
    assert tr._sentences(NO_EVIDENCE) == []


def test_run_uses_fallback_only_on_failure(monkeypatch):
    seen = []

    def fake_ask(tech, q, node, **kw):
        seen.append((tech, q))
        fail = "한계" in q and "KIVI" in q          # KIVI 한계 1번 질문만 실패 → 2번으로
        ans = NO_EVIDENCE if fail else f"{q[:10]} 답[p.3]"
        return {"answer": ans, "evidence": [] if fail else [{"claim": "c", "tag": "논문", "ref": "x", "page": 3}],
                "retrieval_entry": {"tech": tech, "relevance": "no_evidence" if fail else "yes"},
                "citations": [] if fail else [{"id_or_url": tech}], "llm_calls": 3, "faithful": True, "unsupported": []}

    monkeypatch.setattr(tr, "ask", fake_ask)
    out = tr.run({})
    assert set(out) == {"tech_summary", "citations", "retrieval_log", "llm_calls"}
    assert len(seen) == 11 and len(out["retrieval_log"]) == 11      # 10 + 대체 1
    kivi = out["tech_summary"]["KIVI"]
    assert kivi["limitations"] and kivi["numbers"] and kivi["overview"].endswith("[p.3]")
    assert out["llm_calls"] == 33
    assert not any("KIVI" in q and "InfiniGen" in q for _, q in seen)   # 기술별 독립 호출
