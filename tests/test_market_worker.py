"""시장 워커의 격리, 출처, 실패 처리 계약."""
import copy
import json
from types import SimpleNamespace

import pytest

from agents import _web_eval as web
from agents import market
from graph.state import Eval, Evidence, Ref
from tests.web_fakes import claim, install, source, state


def response(payload):
    tech = payload["tech"]
    axis = {"grade": "중", "reason": claim(tech)}
    return {
        "adoption": axis, "market_connection": axis, "ecosystem": axis,
        "positives": [claim(tech)],
        "negatives": [claim(tech, "배포 복잡성", "Complex deployment.")],
        "confidence": 0.8,
    }


def test_market_fixture_contract_and_technology_isolation(monkeypatch):
    queries, prompts = install(monkeypatch, response)
    original = state()
    snapshot = copy.deepcopy(original)
    result = market.run(original)
    assert original == snapshot
    assert set(result) == {"market_eval", "citations", "llm_calls"}
    assert result["llm_calls"] == 2  # 누적값이 아닌 이번 호출의 증가분
    assert len(queries) == 6 and len(prompts) == 2
    for tech, messages in zip(web.TECHS, prompts, strict=True):
        other = "InfiniGen" if tech == "KIVI" else "KIVI"
        assert other not in json.dumps(messages)
        assert "시장성 Rubric" in messages[0][1]
        evaluation = result["market_eval"][tech]
        assert set(evaluation) == set(Eval.__annotations__)
        assert all(name in evaluation["grade"] for name in ("채택", "시장 연결", "생태계"))
        assert all(set(e) == set(Evidence.__annotations__) for e in evaluation["evidence"])
        assert all(f"[웹 {source(tech)['url']}]" in line for line in evaluation["positives"])
    assert len(result["citations"]) == 2
    assert all(set(ref) == set(Ref.__annotations__) for ref in result["citations"])
    assert all(ref["year"] == "2024" for ref in result["citations"])


def test_rejects_invented_url_and_quote_without_trusting_grade(monkeypatch):
    def bad(payload):
        result = response(payload)
        result["adoption"] = {
            "grade": "상", "reason": {"text": "대기업 채택", "source_url": "https://invented.org", "quote": "fake"},
        }
        result["negatives"] = [claim(payload["tech"], "위조 근거", "not present in sources")]
        return result

    install(monkeypatch, bad)
    result = market.run(state())
    for evaluation in result["market_eval"].values():
        assert "채택: 근거 없음" in evaluation["grade"]
        assert evaluation["negatives"] == []
        assert evaluation["confidence"] <= 0.3
        assert "invented.org" not in json.dumps(evaluation)


def test_missing_input_does_not_silently_use_fixtures(monkeypatch):
    queries, prompts = install(monkeypatch, response)
    result = market.run({})
    assert not queries and not prompts and result["llm_calls"] == 0
    assert all(e["grade"] == "평가 불가" for e in result["market_eval"].values())


def test_inference_alone_cannot_assert_market_grade(monkeypatch):
    def inferred(payload):
        result = response(payload)
        result["adoption"] = {"grade": "상", "reason": {
            "text": "공식 도입 여부는 확인되지 않았다", "source_url": None, "quote": "",
        }}
        return result

    install(monkeypatch, inferred)
    result = market.run(state())
    assert all("채택: 근거 없음" in e["grade"] for e in result["market_eval"].values())


@pytest.mark.parametrize("kind", ["empty", "search_error", "llm_error", "parse_error"])
def test_failures_produce_nonempty_state_and_count_attempts(monkeypatch, kind):
    install(monkeypatch, response)

    def fail(*args, **kwargs):
        raise RuntimeError("sensitive error text must not appear in State")

    if kind == "empty":
        monkeypatch.setattr(web, "search_client", lambda: SimpleNamespace(search=lambda **kw: {"results": []}))
    elif kind == "search_error":
        monkeypatch.setattr(web, "search_client", lambda: SimpleNamespace(search=fail))
    elif kind == "llm_error":
        monkeypatch.setattr(web, "generator", lambda schema: SimpleNamespace(invoke=fail))
    else:
        monkeypatch.setattr(web, "generator", lambda schema: SimpleNamespace(invoke=lambda msgs: {}))
    result = market.run(state())
    assert set(result["market_eval"]) == set(web.TECHS)
    assert all(e["confidence"] == 0 and e["rationale"] for e in result["market_eval"].values())
    assert result["llm_calls"] == (2 if kind in {"llm_error", "parse_error"} else 0)
    assert "sensitive error text" not in json.dumps(result)


def test_partial_search_failure_keeps_successful_sources(monkeypatch):
    install(monkeypatch, response)
    count = 0

    def search(**kwargs):
        nonlocal count
        count += 1
        if count % 3 == 1:
            raise TimeoutError()
        return {"results": [source("KIVI" if '"KIVI"' in kwargs["query"] else "InfiniGen")]}

    monkeypatch.setattr(web, "search_client", lambda: SimpleNamespace(search=search))
    result = market.run(state())
    assert result["llm_calls"] == 2 and len(result["citations"]) == 2
    assert all("일부 실패" in e["rationale"] for e in result["market_eval"].values())


def test_only_used_citations_returned_and_no_duplicate_delta(monkeypatch):
    install(monkeypatch, response)
    original = state()
    first = market.run(original)
    original["citations"] = first["citations"]
    assert market.run(original)["citations"] == []
    sources = {source(t)["url"]: source(t) for t in web.TECHS}
    grounding = web.Grounding(sources)
    grounding.claim(web.Claim.model_validate(claim("KIVI")))
    assert [ref["id_or_url"] for ref in grounding.citations()] == [source("KIVI")["url"]]


def test_search_deduplicates_and_combines_snippets(monkeypatch):
    calls = iter(["first quote", "second quote"])
    monkeypatch.setattr(web, "search_client", lambda: SimpleNamespace(
        search=lambda **kwargs: {"results": [{**source("KIVI"), "content": next(calls)}]},
    ))
    sources, notes = web.search(["one", "two"])
    assert len(sources) == 1 and not notes
    assert sources[source("KIVI")["url"]]["content"] == "first quote\nsecond quote"


def test_excerpt_ids_resolve_to_verbatim_source_and_unknown_ids_are_rejected():
    sources = {source("KIVI")["url"]: source("KIVI")}
    grounding = web.Grounding(sources)
    selected = web.Claim.model_validate(claim("KIVI", "배포 복잡성", "E4"))
    assert grounding.quote(selected) == "Complex deployment."
    assert grounding.claim(selected, require_web=True)
    assert grounding.claim(selected.model_copy(update={"quote": "E999"}), require_web=True) is None
    assert grounding.claim(selected.model_copy(update={"source_url": "https://unknown.org"})) is None


def test_indexed_and_literal_quotes_share_deduplication_key():
    grounding = web.Grounding({source("KIVI")["url"]: source("KIVI")})
    claims = [web.Claim.model_validate(claim("KIVI", "비용", "E3")),
              web.Claim.model_validate(claim("KIVI", "같은 비용", "High memory cost."))]
    assert len(grounding.claims(claims, require_web=True)) == 1


def test_excerpts_remain_contiguous_even_for_long_unbroken_content():
    content = "Long paragraph " * 100 + "\n" + "a" * 900
    chunks = web.excerpts(content)
    assert chunks and all(0 < len(text) <= 420 and text in content for text in chunks.values())


def test_live_style_excerpt_ids_produce_market_evidence(monkeypatch):
    def indexed(payload):
        result = response(payload)
        assert "excerpts" in payload["sources"][0]
        assert "content" not in payload["sources"][0]
        for axis in ("adoption", "market_connection", "ecosystem"):
            result[axis]["reason"]["quote"] = "E1"
        return result

    install(monkeypatch, indexed)
    result = market.run(state())
    assert all("채택: 중" in e["grade"] for e in result["market_eval"].values())
