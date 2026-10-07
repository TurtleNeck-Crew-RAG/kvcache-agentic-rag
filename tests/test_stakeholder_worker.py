"""이해관계자 재작업의 기술 격리·근거 보존·중복·종료 검증. [소유: B 심준용]"""
import copy
import json
from types import SimpleNamespace

import pytest

from agents import _web_eval as web
from agents import stakeholder
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
    current.setdefault("stakeholder_eval", {}).update(output["stakeholder_eval"])
    current["llm_calls"] += output["llm_calls"]
    current["citations"] += output["citations"]


def rework(current, tech, attempt, *, gap="negatives", hint="independent criticism"):
    current["rework_request"] = {
        "worker": "stakeholder", "tech": tech, "gap": gap, "hint_query": hint,
    }
    current.setdefault("retry", {})[f"stakeholder:{tech}"] = attempt


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


def test_rework_preserves_evidence_and_runs_only_requested_technology(monkeypatch):
    def respond(payload):
        data = response(payload, negatives=1)
        if payload["mode"] == "negative_only":
            data["positives"] = []
            data["negatives"] = [claim(payload["tech"], "배포 복잡성", "Complex deployment.")]
        return data

    queries, prompts = install(monkeypatch, respond)
    current = ready_state()
    apply(current, stakeholder.run(current))
    original = copy.deepcopy(current["stakeholder_eval"])
    rework(current, "InfiniGen", 1)
    result = stakeholder.run(current)
    assert set(result["stakeholder_eval"]) == {"InfiniGen"}
    assert result["llm_calls"] == 1 and not result["citations"]
    assert len(queries) == 14 and all('"InfiniGen"' in q["query"] for q in queries[10:])
    assert "independent criticism" in queries[10]["query"]
    payload = json.loads(prompts[-1][1][1])
    assert payload["mode"] == "negative_only" and payload["retry"] == 1
    evaluation = result["stakeholder_eval"]["InfiniGen"]
    assert evaluation["positives"] == original["InfiniGen"]["positives"]
    assert len(evaluation["negatives"]) == 2
    assert all(e in evaluation["evidence"] for e in original["InfiniGen"]["evidence"])
    apply(current, result)
    assert current["stakeholder_eval"]["KIVI"] == original["KIVI"]


def test_non_negative_gap_uses_balanced_mode_with_hint(monkeypatch):
    queries, prompts = install(monkeypatch, response)
    current = ready_state()
    apply(current, stakeholder.run(current))
    rework(current, "KIVI", 1, gap="source_diversity", hint="independent analyst adoption")
    result = stakeholder.run(current)
    assert set(result["stakeholder_eval"]) == {"KIVI"}
    assert result["llm_calls"] == 1
    assert len(queries) == 16 and "independent analyst adoption" in queries[10]["query"]
    assert json.loads(prompts[-1][1][1])["mode"] == "balanced"


def test_rework_request_for_another_worker_is_ignored(monkeypatch):
    queries, prompts = install(monkeypatch, response)
    current = ready_state()
    current["rework_request"] = {
        "worker": "market", "tech": "InfiniGen", "gap": "negatives", "hint_query": "ignore me",
    }
    result = stakeholder.run(current)
    assert set(result["stakeholder_eval"]) == set(web.TECHS)
    assert result["llm_calls"] == 2 and len(prompts) == 2 and len(queries) == 10
    assert all("ignore me" not in q["query"] for q in queries)


def test_duplicate_quotes_and_inferences_cannot_satisfy_negative_quota(monkeypatch):
    def respond(payload):
        data = response(payload, 1)
        data["negatives"][0]["text"] += f" (표현 {payload['retry']})"
        data["negatives"] += [
            claim(payload["tech"], "동일 원문의 다른 표현", "High memory cost."),
            {"text": "근거 없는 비판", "source_url": None, "quote": ""},
        ]
        return data

    install(monkeypatch, respond)
    current = ready_state()
    apply(current, stakeholder.run(current))
    for attempt in (1, 2):
        rework(current, "InfiniGen", attempt)
        result = stakeholder.run(current)
        apply(current, result)
        evaluation = current["stakeholder_eval"]["InfiniGen"]
        real = [n for n in evaluation["negatives"] if n != stakeholder.FAILURE]
        assert len(real) == 1
        assert (stakeholder.FAILURE in evaluation["negatives"]) == (attempt == 2)


@pytest.mark.parametrize("failure", ["empty", "exception", "parse"])
def test_failure_preserves_previous_evidence_and_marks_second_rework(monkeypatch, failure):
    install(monkeypatch, lambda p: response(p, 1))
    current = ready_state()
    apply(current, stakeholder.run(current))
    original = copy.deepcopy(current["stakeholder_eval"]["InfiniGen"])
    if failure == "empty":
        monkeypatch.setattr(web, "search_client", lambda: SimpleNamespace(search=lambda **kw: {"results": []}))
    elif failure == "exception":
        def fail(messages):
            raise RuntimeError("do not expose credentials")
        monkeypatch.setattr(web, "generator", lambda schema: SimpleNamespace(invoke=fail))
    else:
        monkeypatch.setattr(web, "generator", lambda schema: SimpleNamespace(invoke=lambda msgs: {}))
    for attempt in (1, 2):
        rework(current, "InfiniGen", attempt)
        result = stakeholder.run(current)
        assert set(result["stakeholder_eval"]) == {"InfiniGen"}
        assert result["llm_calls"] == (0 if failure == "empty" else 1)
        assert "do not expose credentials" not in str(result)
        apply(current, result)
        evaluation = current["stakeholder_eval"]["InfiniGen"]
        assert evaluation["positives"] == original["positives"]
        assert all(e in evaluation["evidence"] for e in original["evidence"])
    assert stakeholder.FAILURE in current["stakeholder_eval"]["InfiniGen"]["negatives"]


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


def test_positive_shortage_is_explicit(monkeypatch):
    def respond(payload):
        data = response(payload)
        data["positives"] = []
        return data

    install(monkeypatch, respond)
    result = stakeholder.run(ready_state())
    assert all("찬성 근거 2건 미만" in e["rationale"] for e in result["stakeholder_eval"].values())


def test_rework_does_not_erase_group_assessment_missing_from_new_search(monkeypatch):
    def respond(payload):
        data = response(payload, 1)
        if payload["mode"] == "negative_only":
            data["investment_media"] = {"stance": "비판", "reason": {
                "text": "새 검색에 미디어 자료 없음", "source_url": None, "quote": "",
            }}
        return data

    install(monkeypatch, respond)
    current = ready_state()
    apply(current, stakeholder.run(current))
    rework(current, "InfiniGen", 1)
    result = stakeholder.run(current)
    assert "투자·미디어: 중립" in result["stakeholder_eval"]["InfiniGen"]["grade"]


def test_failure_marker_is_idempotent_and_never_counts_as_real_evidence():
    result = web.blank("자료 없음")
    stakeholder._finish(result, exhausted=True)
    stakeholder._finish(result, exhausted=True)
    assert result["negatives"] == [stakeholder.FAILURE]


def test_excerpt_ids_preserve_real_quotes_and_deduplicate_across_rework(monkeypatch):
    def indexed(payload):
        data = response(payload, 1)
        data["negatives"][0]["quote"] = "E3" if not payload["retry"] else "High memory cost."
        data["negatives"][0]["text"] = f"비용 표현 {payload['retry']}"
        return data

    install(monkeypatch, indexed)
    current = ready_state()
    apply(current, stakeholder.run(current))
    rework(current, "InfiniGen", 1)
    result = stakeholder.run(current)
    evaluation = result["stakeholder_eval"]["InfiniGen"]
    assert len(evaluation["negatives"]) == 1
    assert "(원문: High memory cost.)" in evaluation["negatives"][0]
