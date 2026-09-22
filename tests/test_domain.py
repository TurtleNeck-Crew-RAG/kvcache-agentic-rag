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

    # 태그 없는 문장은 거부하지 않고 [추론] 으로 집계한다 (거부하면 워커 전체가 실패 기록으로 대체됨)
    data = _evaluation("KIVI").model_dump()
    data["negatives"][0] = "출처가 없는 한계"
    out = domain.DomainEvaluationOutput.model_validate(data)
    assert out.negatives[0].endswith("[추론]")
    # LLM 의 태그 변형은 그대로 인정
    for variant in ("한계 [논문 p.2, p.6]", "한계 [논문 2402.02750 p.2]", "한계 [논문 없음]", "한계 [p.7]", "한계 [웹 https://x]"):
        data["negatives"][0] = variant
        assert domain.DomainEvaluationOutput.model_validate(data).negatives[0] == variant
