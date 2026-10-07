import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

# CI intentionally installs only langgraph + pytest.  Stub the constructor that
# agents._common imports; every test replaces domain.llm before it is called.
langchain_openai = ModuleType("langchain_openai")
langchain_openai.ChatOpenAI = Mock  # type: ignore[attr-defined]
sys.modules.setdefault("langchain_openai", langchain_openai)
langchain_tavily = ModuleType("langchain_tavily")
langchain_tavily.TavilySearch = Mock  # type: ignore[attr-defined]
sys.modules.setdefault("langchain_tavily", langchain_tavily)

import agents.domain as domain  # noqa: E402
import graph.observe as ob  # noqa: E402


@pytest.fixture(autouse=True)
def _no_outputs(monkeypatch, tmp_path):
    """run() 이 검색 로그를 outputs/ 에 쓰지 않게 — 임시 폴더로."""
    monkeypatch.setattr(ob, "OUT", tmp_path)


def _rag_answer(tech: str, question: str, node: str) -> dict:
    assert node == "domain"
    ref = "2402.02750" if tech == "KIVI" else "2406.19707"
    return {
        "answer": f"{tech}: {question} [p.1]",
        "evidence": [{"claim": question, "tag": "논문", "ref": ref, "page": 1}],
        "retrieval_entry": {
            "node": node,
            "tech": tech,
            "query_before": question,
            "query_after": None,
            "hits_before": [f"{tech}-1"],
            "hits_after": None,
            "relevance": "yes",
            "rewritten": False,
        },
        "llm_calls": 2,
    }


def _evaluation(tech: str) -> domain.DomainEvaluationOutput:
    ref = "2402.02750" if tech == "KIVI" else "2406.19707"
    evidence = [{"claim": "paper fact", "tag": "논문", "ref": ref, "page": 1}]
    if tech == "InfiniGen":
        evidence.append(
            {
                "claim": "mobile memory counter example",
                "tag": "웹",
                "ref": "https://example.com/mobile-memory",
                "page": None,
            }
        )
    return domain.DomainEvaluationOutput(
        verdict="조건부",
        rationale="배포 제약의 일부 조건을 추가로 확인해야 한다 [추론].",
        positives=["재학습 없이 적용할 수 있다 [논문 p.1]."],
        negatives=[
            "대상 기기의 대역폭에서는 검증되지 않았다 [논문 p.1].",
            "모바일 소비 전력은 확인되지 않았다 [추론].",
        ],
        axes={
            "recall": "정확도 결과를 임계값 없이 보고한다 [논문 p.1].",
            "latency": "추가 오버헤드를 보고한다 [논문 p.1].",
            "memory": "KV cache 메모리 절감 효과를 보고한다 [논문 p.1].",
        },
        evidence=evidence,
        confidence=0.7,
    )


def test_run_extracts_five_facts_per_tech_and_returns_state_contract(monkeypatch):
    prompts: list[str] = []

    class FakeEvaluator:
        def invoke(self, prompt: str):
            prompts.append(prompt)
            tech = "KIVI" if "## 평가 기술\nKIVI" in prompt else "InfiniGen"
            return _evaluation(tech)

    fake_llm = Mock()
    fake_llm.with_structured_output.return_value = FakeEvaluator()
    monkeypatch.setattr(domain, "llm", lambda role: fake_llm)
    monkeypatch.setattr(domain, "ask", _rag_answer)

    search_queries: list[str] = []

    class FakeSearch:
        def __init__(self, **kwargs):
            assert kwargs == {
                "max_results": 3,
                "search_depth": "advanced",
                "include_answer": False,
            }

        def invoke(self, request: dict[str, str]):
            search_queries.append(request["query"])
            return {
                "results": [
                    {
                        "title": "Mobile memory hierarchy",
                        "url": "https://example.com/mobile-memory",
                        "content": "A mobile memory offloading approach is evaluated.",
                        "score": 0.9,
                    }
                ]
            }

    monkeypatch.setattr(domain, "TavilySearch", FakeSearch)

    state = {
        "trace_id": "t-domain",
        "domain": {
            "constraints": {
                "memory": {"ram_gb": [8, 16]},
                "deployment": {"retraining": False},
            },
            "axes": [{"key": "recall"}, {"key": "latency"}, {"key": "memory"}],
            "counter_examples": ["LLM in a flash", "Samsung LPDDR5X-PIM"],
        },
        "selected": {
            "sw": {
                "name": "KIVI",
                "authors": "Liu, Z., Yuan, J., Jin, H. et al.",
                "paper": "KIVI paper",
                "arxiv": "2402.02750",
                "venue": "ICML 2024",
            },
            "hw": {
                "name": "InfiniGen",
                "authors": "Lee, W., Lee, J., Seo, J., Sim, J.",
                "paper": "InfiniGen paper",
                "arxiv": "2406.19707",
                "venue": "OSDI 2024",
            },
        },
    }
    result = domain.run(state)

    assert set(result) == {"domain_eval", "citations", "retrieval_log", "llm_calls"}
    assert set(result["domain_eval"]) == {"KIVI", "InfiniGen"}
    assert len(result["retrieval_log"]) == 10
    logged = ob.read_jsonl(ob.RETRIEVAL, "t-domain")          # 정본은 파일 — State 반환은 이행 중(#62)
    assert len(logged) == 10 and {r["node"] for r in logged} == {"domain"}
    assert [r["query_before"] for r in logged] == [e["query_before"] for e in result["retrieval_log"]]
    assert ob.retrieval_entries({"trace_id": "t-domain"}) == logged
    assert result["llm_calls"] == 22
    assert len(result["citations"]) == 3
    assert result["citations"][0]["authors"] == "Liu, Z., Yuan, J., Jin, H. et al."
    assert result["citations"][2]["id_or_url"] == "https://example.com/mobile-memory"
    assert result["domain_eval"]["KIVI"]["grade"] == "조건부"
    assert set(result["domain_eval"]["KIVI"]["axes"]) == {"recall", "latency", "memory"}

    assert len(prompts) == 2
    assert "InfiniGen" not in prompts[0]
    assert "KIVI" not in prompts[1]
    assert "https://example.com/mobile-memory" not in prompts[0]
    assert "https://example.com/mobile-memory" in prompts[1]
    assert len(search_queries) == 2


def test_run_validates_dict_structured_output(monkeypatch):
    class FakeEvaluator:
        def invoke(self, prompt: str):
            tech = "KIVI" if "## 평가 기술\nKIVI" in prompt else "InfiniGen"
            return _evaluation(tech).model_dump()

    fake_llm = Mock()
    fake_llm.with_structured_output.return_value = FakeEvaluator()
    monkeypatch.setattr(domain, "llm", lambda role: fake_llm)
    monkeypatch.setattr(domain, "ask", _rag_answer)
    monkeypatch.setattr(domain, "TavilySearch", Mock())

    result = domain.run({"domain": {"constraints": {}}})

    assert result["domain_eval"]["InfiniGen"]["verdict"] == "조건부"


def test_run_rejects_missing_domain_config():
    with pytest.raises(ValueError, match="state\\['domain'\\]"):
        domain.run({})


def test_fact_questions_do_not_prime_domain_verdict():
    assert all(
        forbidden not in question
        for question in domain.FACT_QUESTIONS
        for forbidden in ("온디바이스", "적합")
    )


def test_domain_output_requires_two_tagged_negatives():
    data = _evaluation("KIVI").model_dump()
    data["negatives"] = data["negatives"][:1]
    with pytest.raises(ValueError, match="at least 2"):
        domain.DomainEvaluationOutput.model_validate(data)

    data = _evaluation("KIVI").model_dump()
    data["negatives"][0] = "출처가 없는 한계"
    out = domain.DomainEvaluationOutput.model_validate(data)        # 무태그 문장은 버리지 않고 [추론] 으로 기록 (#40 후속)
    assert out.negatives[0] == "출처가 없는 한계 [추론]"


# ── #81: rework_request — 요청 기술만 보강 ─────────────────────────────

from graph.state import merge_by_tech  # noqa: E402

WEB_URL = "https://example.com/mobile-memory"


class _Harness:
    """ask · TavilySearch · LLM 을 기록하는 가짜 묶음."""

    def __init__(self, monkeypatch, web_tech: set[str] | None = None):
        self.asks: list[tuple[str, str]] = []
        self.queries: list[str] = []
        self.prompts: list[str] = []
        web_tech = web_tech if web_tech is not None else {"InfiniGen"}
        harness = self

        def fake_ask(tech, question, node):
            harness.asks.append((tech, question))
            return _rag_answer(tech, question, node)

        class FakeSearch:
            def __init__(self, **kwargs):
                pass

            def invoke(self, request):
                harness.queries.append(request["query"])
                return {"results": [{"title": "Mobile memory", "url": WEB_URL, "content": "counter", "score": 0.9}]}

        class FakeEvaluator:
            def invoke(self, prompt):
                harness.prompts.append(prompt)
                tech = "KIVI" if "## 평가 기술\nKIVI" in prompt else "InfiniGen"
                out = _evaluation(tech)
                if tech in web_tech and tech == "KIVI":
                    out.evidence.append(domain.EvidenceOutput(claim="KIVI mobile counter", tag="웹", ref=WEB_URL))
                return out

        fake_llm = Mock()
        fake_llm.with_structured_output.return_value = FakeEvaluator()
        monkeypatch.setattr(domain, "llm", lambda role: fake_llm)
        monkeypatch.setattr(domain, "ask", fake_ask)
        monkeypatch.setattr(domain, "TavilySearch", FakeSearch)


def _previous_eval(tech: str) -> dict:
    data = domain._as_domain_eval(_evaluation(tech))
    data["evidence"] = data["evidence"] + [{"claim": "earlier fact", "tag": "논문", "ref": "prev-ref", "page": 3}]
    data["rationale"] = f"{tech} 직전 판정 근거 [논문 p.3]."
    return data


def _rework_state(worker="domain", tech="KIVI", gap="evidence", hint="KIVI 스마트폰 메모리 절감 조건") -> dict:
    return {
        "trace_id": "t-rework",
        "domain": {"constraints": {"memory": {"ram_gb": [8, 16]}},
                   "counter_examples": ["LLM in a flash", "Samsung LPDDR5X-PIM"]},
        "selected": {"sw": {"name": "KIVI", "arxiv": "2402.02750"}, "hw": {"name": "InfiniGen", "arxiv": "2406.19707"}},
        "domain_eval": {t: _previous_eval(t) for t in ("KIVI", "InfiniGen")},
        "rework_request": {"worker": worker, "tech": tech, "gap": gap, "hint_query": hint},
    }


def test_other_worker_rework_keeps_full_run(monkeypatch):
    h = _Harness(monkeypatch)

    result = domain.run(_rework_state(worker="market"))

    assert set(result["domain_eval"]) == {"KIVI", "InfiniGen"}
    assert len(h.asks) == 10 and len(h.prompts) == 2
    assert not any("## 재작업 요청" in p for p in h.prompts)


def test_paper_gap_reinforces_only_requested_tech_with_hint(monkeypatch):
    h = _Harness(monkeypatch)
    state = _rework_state(gap="evidence", hint="KIVI 스마트폰 메모리 절감 조건")

    result = domain.run(state)

    assert set(result["domain_eval"]) == {"KIVI"}
    assert h.asks == [("KIVI", "KIVI 스마트폰 메모리 절감 조건")]       # 사실 질의 5개를 다시 돌리지 않는다
    assert h.queries == []                                               # 논문 gap 은 웹 검색 없음
    assert len(h.prompts) == 1 and "InfiniGen" not in h.prompts[0]
    assert "## 재작업 요청" in h.prompts[0] and "KIVI 직전 판정 근거" in h.prompts[0]
    assert result["llm_calls"] == 2 + 1
    assert len(result["retrieval_log"]) == 1
    claims = {e["claim"] for e in result["domain_eval"]["KIVI"]["evidence"]}
    assert {"paper fact", "earlier fact"} <= claims                      # 기존 근거 유지
    assert set(result["domain_eval"]["KIVI"]["axes"]) == {"recall", "latency", "memory"}
    assert result["domain_eval"]["KIVI"]["grade"] == result["domain_eval"]["KIVI"]["verdict"]
    assert not ({"next", "retry", "sufficiency", "eval_result", "rework_request"} & result.keys())


def test_reducer_merge_preserves_other_tech(monkeypatch):
    _Harness(monkeypatch)
    state = _rework_state()

    result = domain.run(state)
    merged = merge_by_tech(state["domain_eval"], result["domain_eval"])

    assert merged["InfiniGen"] is state["domain_eval"]["InfiniGen"]
    assert merged["KIVI"] is result["domain_eval"]["KIVI"]


def test_counter_example_gap_searches_web_for_kivi(monkeypatch):
    h = _Harness(monkeypatch, web_tech={"KIVI"})
    hint = "KIVI 스마트폰 온디바이스 한계 반례 메모리 대역폭 전력"

    result = domain.run(_rework_state(tech="KIVI", gap="counter_example", hint=hint))

    assert h.asks == []                                                  # 반례 gap 은 논문 RAG 로 못 메운다
    assert h.queries == [f"KIVI {hint}"]
    assert "## HW 반례 웹 검색 결과" in h.prompts[0] and WEB_URL in h.prompts[0]
    assert [c["id_or_url"] for c in result["citations"] if c["type"] == "웹"] == [WEB_URL]
    assert result["llm_calls"] == 1


def test_counter_example_gap_for_infinigen_adds_hint_to_counter_examples(monkeypatch):
    h = _Harness(monkeypatch)

    domain.run(_rework_state(tech="InfiniGen", gap="counter_example", hint="InfiniGen 모바일 반례"))

    assert len(h.queries) == 3 and h.queries[-1] == "InfiniGen InfiniGen 모바일 반례"
    assert any("LLM in a flash" in q for q in h.queries)


def test_missing_previous_result_reruns_that_tech_from_scratch(monkeypatch):
    h = _Harness(monkeypatch)
    state = _rework_state(tech="InfiniGen", gap="missing")
    state["domain_eval"].pop("InfiniGen")

    result = domain.run(state)

    assert set(result["domain_eval"]) == {"InfiniGen"}
    assert [t for t, _ in h.asks] == ["InfiniGen"] * 5
    assert "## 재작업 요청" not in h.prompts[0]
    assert h.queries[-1] == "InfiniGen KIVI 스마트폰 메모리 절감 조건"   # hint 도 보강 검색에 쓴다


def test_multiline_evaluator_feedback_is_clipped_to_first_line(monkeypatch):
    h = _Harness(monkeypatch)
    feedback = "groundedness: 「KIVI…」 근거 없는 단정 — 수치 출처를 다시 조사한다 (→ domain:KIVI)\nneutrality: …"

    domain.run(_rework_state(gap="eval", hint=feedback))

    assert h.asks == [("KIVI", feedback.splitlines()[0][: domain.MAX_HINT_CHARS])]


def test_unknown_rework_tech_is_rejected(monkeypatch):
    _Harness(monkeypatch)

    with pytest.raises(ValueError, match="unknown domain rework technology"):
        domain.run(_rework_state(tech="H2O"))
