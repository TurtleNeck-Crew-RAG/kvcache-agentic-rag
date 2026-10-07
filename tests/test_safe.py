"""graph/safe.py — 노드 예외가 자기 키의 실패 기록 + node_status · last_error · errors 로 바뀌는지. LLM 불필요."""
from graph.safe import fallback, safe
from graph.state import init_state


def _boom(state):
    raise NotImplementedError("stakeholder: 구현 예정")


def test_safe_returns_fallback_on_exception(capsys):
    out = safe("stakeholder", _boom)(init_state({}, {}))
    assert set(out) == {"stakeholder_eval", "node_status", "last_error", "errors"}
    assert out["node_status"] == {"stakeholder": "failed"}          # fallback 이 정상 결과처럼 보이지 않게
    assert out["last_error"]["node"] == "stakeholder" and out["last_error"]["type"] == "NotImplementedError"
    assert out["errors"] == [out["last_error"]]
    for tech in ("KIVI", "InfiniGen"):
        e = out["stakeholder_eval"][tech]
        assert e["grade"] == "근거 없음" and "NotImplementedError" in e["rationale"] and e["negatives"] == []
    assert "[safe] stakeholder 실패" in capsys.readouterr().err


def test_safe_passes_through_normal_output_and_marks_ok():
    out = safe("market", lambda s: {"market_eval": {"KIVI": {}}})({})
    assert out == {"market_eval": {"KIVI": {}}, "node_status": {"market": "ok"}}


def test_fallback_during_rework_touches_only_that_tech():
    """rework_request 가 자기 것이면 그 기술만 실패 기록 — merge_by_tech 라 다른 기술의 정상 결과가 남는다."""
    s = {"rework_request": {"worker": "market", "tech": "InfiniGen", "gap": "", "hint_query": ""}}
    out = safe("market", _boom)(s)
    assert set(out["market_eval"]) == {"InfiniGen"}


def test_judge_node_failure_has_no_payload():
    out = safe("evaluator", _boom)({})
    assert set(out) == {"node_status", "last_error", "errors"} and out["node_status"] == {"evaluator": "failed"}


def test_fallback_keys_match_state_for_every_worker():
    keys = {
        "tech_research": {"tech_summary"},
        "market": {"market_eval"},
        "stakeholder": {"stakeholder_eval"},
        "domain": {"domain_eval"},
        "synthesis": {"synthesis", "trl_estimate", "neutrality"},
        "report": {"report_md"},
    }
    for name, expect in keys.items():
        assert set(fallback(name, "x")) == expect, name
    assert fallback("domain", "x")["domain_eval"]["KIVI"]["verdict"] == "부적합"
    assert fallback("synthesis", "x")["neutrality"]["result"] == "pass"     # 3' 루프에 걸리지 않는다


def test_safe_reason_is_single_short_line():
    def boom(state):
        raise ValueError("6 validation errors for X\nrationale\n  Value error, 판단 문장에 출처 태그가 필요합니다 …\n" + "x" * 500)

    e = safe("domain", boom)({})["domain_eval"]["KIVI"]
    assert "\n" not in e["rationale"] and len(e["rationale"]) < 200
    assert e["rationale"].startswith("워커 실패 — ValueError: 6 validation errors for X")


def _fake_llm(total: int):
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    usage = {"input_tokens": total - 1, "output_tokens": 1, "total_tokens": total}
    return GenericFakeChatModel(messages=iter([AIMessage(content="x", response_metadata={"model_name": "fake"},
                                                         usage_metadata=usage)]))


def test_safe_counts_tokens_of_llm_calls_inside_node():
    """#102 — 노드 안 LLM 토큰을 safe 가 집계해 tokens 로 반환 (워커 코드 수정 없음)."""
    def worker(state):
        _fake_llm(100).invoke("a")
        _fake_llm(23).invoke("b")
        return {"market_eval": {"KIVI": {}}, "llm_calls": 2}

    out = safe("market", worker)({})
    assert out["tokens"] == 123 and out["llm_calls"] == 2 and out["node_status"] == {"market": "ok"}


def test_safe_counts_tokens_spent_before_failure_and_omits_zero():
    def worker(state):
        _fake_llm(50).invoke("a")
        raise RuntimeError("boom")

    out = safe("market", worker)({})
    assert out["tokens"] == 50 and out["node_status"] == {"market": "failed"}
    assert "tokens" not in safe("market", lambda s: {"market_eval": {}})({})      # LLM 없으면 키 없음
