"""이해관계자 재검색의 근거 보존·중복·종료 검증. [소유: B 심준용]"""
import copy
import json
from types import SimpleNamespace

import pytest

from agents import _web_eval as web
from agents import stakeholder
from graph.dispatcher import dispatcher
from graph.state import Eval
from tests.web_fakes import claim, install, source, state


def response(payload, negatives=2):
    tech = payload["tech"]
    group = {"stance": "중립", "reason": claim(tech)}
    return {
        "competitors": group, "developers": group, "investment_media": group,
        "positives": [claim(tech), claim(tech, "커뮤니티 활용", "Community adoption.")],
        "negatives": [claim(tech, "메모리 비용", "High memory cost."),
                      claim(tech, "배포 복잡성", "Complex deployment.")][:negatives],
        "confidence": 0.8,
    }


def apply(current, output):
    current["stakeholder_eval"] = output["stakeholder_eval"]
    current["llm_calls"] += output["llm_calls"]
    current["citations"] += output["citations"]


def ready_state():
    current = state()
    current["market_eval"] = current["domain_eval"] = {"ready": True}
    return current


def test_balanced_contract_and_other_workers_not_passed(monkeypatch):
    queries, prompts = install(monkeypatch, response)
    current = ready_state()
    current["market_eval"] = {"secret": "OTHER_WORKER_MUST_NOT_LEAK"}
    snapshot = copy.deepcopy(current)
    result = stakeholder.run(current)
    assert current == snapshot
    assert set(result) == {"stakeholder_eval", "citations", "llm_calls"}
    assert result["llm_calls"] == 2 and len(queries) == 10
    assert any("LLM in a flash" in q["query"] and "benefits" in q["query"] for q in queries)
    for tech, prompt in zip(web.TECHS, prompts, strict=True):
        payload = json.loads(prompt[1][1])
        assert payload["mode"] == "balanced" and payload["previous_eval"] == {}
        assert "OTHER_WORKER_MUST_NOT_LEAK" not in str(prompt)
        other = "InfiniGen" if tech == "KIVI" else "KIVI"
        assert other not in str(prompt)
        evaluation = result["stakeholder_eval"][tech]
        assert set(evaluation) == set(Eval.__annotations__)
        assert len(evaluation["positives"]) == len(evaluation["negatives"]) == 2
    apply(current, result)
    assert dispatcher(current)["next"] == ["synthesis"]


def test_retry_preserves_positives_and_accumulates_new_negatives(monkeypatch):
    def respond(payload):
        data = response(payload, negatives=1)
        if payload["retry"]:
            data["positives"] = []
            data["negatives"] = [claim(payload["tech"], "배포 복잡성", "Complex deployment.")]
        return data

    queries, prompts = install(monkeypatch, respond)
    current = ready_state()
    apply(current, stakeholder.run(current))
    original = copy.deepcopy(current["stakeholder_eval"])
    decision = dispatcher(current)
    assert decision == {"next": ["stakeholder"], "retry": {"stake": 1}}
    current.update(decision)
    result = stakeholder.run(current)
    assert result["llm_calls"] == 2 and not result["citations"]
    for tech in web.TECHS:
        assert result["stakeholder_eval"][tech]["positives"] == original[tech]["positives"]
        assert len(result["stakeholder_eval"][tech]["negatives"]) == 2
        assert all(e in result["stakeholder_eval"][tech]["evidence"] for e in original[tech]["evidence"])
    assert all("criticism" in q["query"] for q in queries[10:])
    assert all(json.loads(p[1][1])["mode"] == "negative_only" for p in prompts[2:])
    apply(current, result)
    assert dispatcher(current)["next"] == ["synthesis"]


def test_retry_only_searches_technology_with_missing_negatives(monkeypatch):
    queries, prompts = install(monkeypatch, lambda p: response(p, 2 if p["tech"] == "KIVI" else 1))
    current = ready_state()
    apply(current, stakeholder.run(current))
    before = copy.deepcopy(current["stakeholder_eval"]["KIVI"])
    current.update(dispatcher(current))
    result = stakeholder.run(current)
    assert result["llm_calls"] == 1 and len(queries) == 13
    assert result["stakeholder_eval"]["KIVI"] == before
    assert json.loads(prompts[-1][1][1])["tech"] == "InfiniGen"


def test_duplicate_quotes_and_inferences_cannot_satisfy_negative_quota(monkeypatch):
    def respond(payload):
        data = response(payload, 1)
        # 매 재시도에서 표현을 바꿔도 동일한 원문이면 같은 근거다.
        data["negatives"][0]["text"] += f" (표현 {payload['retry']})"
        data["negatives"] += [
            claim(payload["tech"], "동일 원문의 다른 표현", "High memory cost."),
            {"text": "근거 없는 비판", "source_url": None, "quote": ""},
        ]
        return data

    install(monkeypatch, respond)
    current = ready_state()
    for attempt in range(3):
        result = stakeholder.run(current)
        apply(current, result)
        for evaluation in current["stakeholder_eval"].values():
            real = [n for n in evaluation["negatives"] if n != stakeholder.FAILURE]
            assert len(real) == 1
            assert (stakeholder.FAILURE in evaluation["negatives"]) == (attempt == 2)
        decision = dispatcher(current)
        if attempt < 2:
            assert decision["next"] == ["stakeholder"]
        else:
            assert decision["next"] == ["synthesis"]
        current.update(decision)


@pytest.mark.parametrize("failure", ["empty", "exception", "parse"])
def test_failure_preserves_previous_evidence_and_stops_after_two_retries(monkeypatch, failure):
    install(monkeypatch, lambda p: response(p, 1))
    current = ready_state()
    apply(current, stakeholder.run(current))
    original = copy.deepcopy(current["stakeholder_eval"])
    if failure == "empty":
        monkeypatch.setattr(web, "search_client", lambda: SimpleNamespace(search=lambda **kw: {"results": []}))
    elif failure == "exception":
        def fail(messages):
            raise RuntimeError("do not expose credentials")
        monkeypatch.setattr(web, "generator", lambda schema: SimpleNamespace(invoke=fail))
    else:
        monkeypatch.setattr(web, "generator", lambda schema: SimpleNamespace(invoke=lambda msgs: {}))
    for retry in (1, 2):
        current.update(dispatcher(current))
        assert current["retry"]["stake"] == retry
        result = stakeholder.run(current)
        assert result["llm_calls"] == (0 if failure == "empty" else 2)
        assert "do not expose credentials" not in str(result)
        apply(current, result)
        for tech in web.TECHS:
            assert current["stakeholder_eval"][tech]["positives"] == original[tech]["positives"]
            assert all(e in current["stakeholder_eval"][tech]["evidence"] for e in original[tech]["evidence"])
    assert dispatcher(current)["next"] == ["synthesis"]


def test_all_searches_empty_leave_failure_marker_without_inventing_sources(monkeypatch):
    install(monkeypatch, response)
    monkeypatch.setattr(web, "search_client", lambda: SimpleNamespace(search=lambda **kw: {"results": []}))
    current = ready_state()
    for retry in range(3):
        result = stakeholder.run(current)
        assert result["citations"] == [] and result["llm_calls"] == 0
        apply(current, result)
        for e in result["stakeholder_eval"].values():
            assert e["negatives"] == ([stakeholder.FAILURE] if retry == 2 else [])
        current.update(dispatcher(current))
    assert current["next"] == ["synthesis"]


def test_budget_exhaustion_marks_failure_for_both_technologies(monkeypatch):
    install(monkeypatch, lambda p: response(p, 0))
    current = ready_state()
    current["llm_calls"] = 100
    result = stakeholder.run(current)
    assert result["llm_calls"] == 2
    assert all(e["negatives"] == [stakeholder.FAILURE] for e in result["stakeholder_eval"].values())
    apply(current, result)
    assert dispatcher(current)["next"] == ["synthesis"]


def test_missing_summary_records_failure_without_api_calls(monkeypatch):
    queries, prompts = install(monkeypatch, response)
    result = stakeholder.run({})
    assert not queries and not prompts and result["llm_calls"] == 0
    assert all(e["rationale"] and e["confidence"] == 0 for e in result["stakeholder_eval"].values())


def test_fabricated_url_does_not_become_negative_or_citation(monkeypatch):
    def respond(payload):
        data = response(payload, 0)
        data["negatives"] = [{"text": "위조", "source_url": "https://invented.org", "quote": "fake"}]
        return data

    install(monkeypatch, respond)
    result = stakeholder.run(state())
    assert all(not e["negatives"] for e in result["stakeholder_eval"].values())
    assert all(ref["id_or_url"] in {source(t)["url"] for t in web.TECHS} for ref in result["citations"])


def test_positive_shortage_is_explicit_but_does_not_add_unplanned_retry(monkeypatch):
    def respond(payload):
        data = response(payload)
        data["positives"] = []
        return data

    install(monkeypatch, respond)
    current = ready_state()
    result = stakeholder.run(current)
    assert all("찬성 근거 2건 미만" in e["rationale"] for e in result["stakeholder_eval"].values())
    apply(current, result)
    assert dispatcher(current)["next"] == ["synthesis"]


def test_retry_does_not_erase_group_assessment_missing_from_new_search(monkeypatch):
    def respond(payload):
        data = response(payload, 1)
        if payload["retry"]:
            data["investment_media"] = {"stance": "비판", "reason": {
                "text": "새 검색에 미디어 자료 없음", "source_url": None, "quote": "",
            }}
        return data

    install(monkeypatch, respond)
    current = ready_state()
    apply(current, stakeholder.run(current))
    current.update(dispatcher(current))
    result = stakeholder.run(current)
    assert all("투자·미디어: 중립" in e["grade"] for e in result["stakeholder_eval"].values())


def test_failure_marker_is_idempotent_and_never_counts_as_real_evidence():
    result = web.blank("자료 없음")
    stakeholder._finish(result, exhausted=True)
    stakeholder._finish(result, exhausted=True)
    assert result["negatives"] == [stakeholder.FAILURE]
