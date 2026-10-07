"""assess — 결정론 층 규칙 · 재판정 범위 · Judge 결합. LLM 은 monkeypatch."""
import json
from pathlib import Path

import graph.sufficiency as su

FIX = Path(__file__).parent / "fixtures"


def ev(ref, n=1, tag="웹"):
    return [{"claim": f"c{i}", "tag": tag, "ref": ref, "page": None} for i in range(n)]


def eval_(evidence, negatives=2, grade="중"):
    return {"grade": grade, "rationale": "r", "positives": ["p"], "negatives": [f"n{i}" for i in range(negatives)],
            "evidence": evidence, "confidence": 0.5}


GOOD = eval_(ev("https://a.com/x", 2) + ev("https://b.com/y", 2))


def test_source_key_merges_arxiv_url_and_paper_id():
    assert su.source_key("https://arxiv.org/abs/2402.02750v2") == su.source_key("2402.02750") == "arXiv:2402.02750"
    assert su.source_key("https://www.github.com/x/y") == "github.com"
    assert su.source_key("") == ""


def test_rules_pass_good_web_cell():
    assert su.rule_check("market", "KIVI", GOOD, {})["rule"] == "pass"


def test_failed_worker_is_always_insufficient_even_with_good_payload():
    v = su.rule_check("market", "KIVI", GOOD, {"market": "failed"})
    assert (v["rule"], v["gap"], v["judge"]) == ("fail", "failed", None)


def test_safe_fallback_record_is_not_evidence():
    v = su.rule_check("market", "KIVI", {**GOOD, "rationale": "워커 실패 — Timeout"}, {})
    assert v["gap"] == "failed"


def test_mostly_no_evidence_grade_fails_but_partial_passes():
    assert su.rule_check("market", "KIVI", {**GOOD, "grade": "채택: 근거 없음 / 시장 연결: 근거 없음 / 생태계: 상"}, {})["gap"] == "no_evidence"
    assert su.rule_check("market", "KIVI", {**GOOD, "grade": "채택: 근거 없음 / 시장 연결: 중 / 생태계: 상"}, {})["rule"] == "pass"


def test_negatives_at_least_two_for_every_perspective():
    one = eval_(GOOD["evidence"], negatives=1)
    for w in ("market", "stakeholder", "domain"):
        v = su.rule_check(w, "KIVI", one, {})
        assert v["gap"] == "negatives" and "KIVI" in v["hint_query"]


def test_web_source_majority_fails_but_half_passes():
    half = eval_(ev("https://a.com", 2) + ev("https://b.com", 1) + ev("https://c.com", 1))      # a.com 50%
    assert su.rule_check("market", "KIVI", half, {})["rule"] == "pass"
    major = eval_(ev("https://a.com", 3) + ev("https://b.com", 2))                              # a.com 60%
    v = su.rule_check("stakeholder", "KIVI", major, {})
    assert v["gap"] == "source_bias" and "과반" in v["reason"]


def test_single_source_dominance_fails():
    # 지난 실행 InfiniGen 시장 근거 8건 중 7건이 github.com — 편중
    biased = eval_(ev("https://github.com/snu-comparch/InfiniGen", 7) + ev("https://medium.com/x", 1))
    v = su.rule_check("market", "InfiniGen", biased, {})
    assert v["gap"] == "source_bias" and "github.com" in v["reason"]


def test_fixture_evals_without_evidence_fail_rules():
    evals = json.loads((FIX / "evals.json").read_text(encoding="utf-8"))
    rules = su.check_rules(evals)
    assert set(rules) == {f"{w}:{t}" for w in ("market", "stakeholder", "domain") for t in ("KIVI", "InfiniGen")}
    assert all(v["rule"] == "fail" and v["gap"] == "evidence" for v in rules.values())


def test_tech_summary_rules_skip_source_diversity():
    ts = {"overview": "o[p.1]", "mechanism": "m", "numbers": ["2.6×[p.1]"], "limitations": ["l[p.9]"],
          "apply_conditions": [], "evidence": ev("2402.02750", 5, tag="논문")}
    assert su.rule_check("tech_research", "KIVI", ts, {})["rule"] == "pass"
    v = su.rule_check("tech_research", "KIVI", {**ts, "limitations": []}, {})
    assert v["gap"] == "limitations" and v["hint_query"].startswith("KIVI")


def test_cells_to_judge_only_new_and_just_ran_and_rework_tech():
    rules = {c: {"rule": "pass"} for c in ("market:KIVI", "market:InfiniGen", "domain:KIVI")}
    prev = {"market:KIVI": {}, "market:InfiniGen": {}, "domain:KIVI": {}}
    st = {"sufficiency": prev, "next": "market", "rework_request": {"worker": "market", "tech": "InfiniGen"}}
    assert su.cells_to_judge(st, rules) == ["market:InfiniGen"]
    st = {"sufficiency": prev, "next": "market", "rework_request": None}
    assert su.cells_to_judge(st, rules) == ["market:KIVI", "market:InfiniGen"]
    assert su.cells_to_judge({"sufficiency": {}, "next": "domain"}, rules) == list(rules)


def test_assess_calls_judge_only_for_rule_passed_cells(monkeypatch):
    calls = []

    def fake_judge(worker, tech, payload):
        calls.append((worker, tech))
        return su.SufficiencyJudgment(sufficient=tech == "KIVI", gap="unsupported", hint_query=f"{tech} 근거", reason="r")

    monkeypatch.setattr(su, "judge_cell", fake_judge)
    st = {"market_eval": {"KIVI": GOOD, "InfiniGen": eval_(ev("https://a.com", 1))}, "next": "market"}
    out = su.assess(st)
    assert calls == [("market", "KIVI")] and out["llm_calls"] == 1
    k, i = out["sufficiency"]["market:KIVI"], out["sufficiency"]["market:InfiniGen"]
    assert (k["rule"], k["judge"], k["hint_query"]) == ("pass", "sufficient", "")
    assert (i["rule"], i["judge"], i["gap"]) == ("fail", None, "evidence")
    assert "next" not in out and "retry" not in out          # 게이트 전용 키는 쓰지 않는다
    assert su.is_sufficient(k) and not su.is_sufficient(i)


def test_judge_insufficient_and_judge_error(monkeypatch):
    monkeypatch.setattr(su, "judge_cell", lambda *a: su.SufficiencyJudgment(
        sufficient=False, gap="off_topic", hint_query="KIVI 도입 기업", reason="시장 근거가 논문 수치뿐"))
    v = su.assess({"market_eval": {"KIVI": GOOD}, "next": "market"})["sufficiency"]["market:KIVI"]
    assert (v["judge"], v["gap"], v["hint_query"]) == ("insufficient", "off_topic", "KIVI 도입 기업")

    def boom(*a):
        raise TimeoutError("judge down")

    monkeypatch.setattr(su, "judge_cell", boom)
    v = su.assess({"market_eval": {"KIVI": GOOD}, "next": "market"})["sufficiency"]["market:KIVI"]
    assert v["judge"] is None and "Judge 실패" in v["reason"] and su.is_sufficient(v)


def test_prompt_renders_with_all_placeholders():
    text = su.render_prompt("sufficiency_judge", tech="KIVI", perspective="시장성", content="x")
    assert "{" + "tech}" not in text and "KIVI" in text and "시장성" in text


def test_domain_needs_web_counter_example_not_share():
    paper = ev("2406.19707", 4, tag="논문")
    assert su.rule_check("domain", "InfiniGen", eval_(paper + ev("https://a.com", 1)), {})["rule"] == "pass"   # 논문 80% 여도 통과
    v = su.rule_check("domain", "KIVI", eval_(paper), {})
    assert v["gap"] == "counter_example" and "반례" in v["hint_query"]


def test_inference_tagged_evidence_is_not_counted():
    # 웹 1건 + [추론] 2건 — 건수로는 3 이지만 출처 있는 근거는 1건
    padded = eval_(ev("https://a.com", 1) + ev("", 2, tag="추론"))
    v = su.rule_check("market", "KIVI", padded, {})
    assert v["gap"] == "evidence" and "1건" in v["reason"]
    ts = {"overview": "o", "mechanism": "m", "numbers": ["n"], "limitations": ["l"], "apply_conditions": [],
          "evidence": ev("2402.02750", 2, tag="논문") + [{"claim": "Faithfulness 미통과", "tag": "추론", "ref": "judge", "page": None}]}
    assert su.rule_check("tech_research", "KIVI", ts, {})["gap"] == "evidence"


def test_failure_marker_in_negatives_is_not_counted():
    # stakeholder 는 재작업 상한 후 "반대 근거 확보 실패 [추론]" 을 붙인다 (#73) — 이건 반대 근거가 아니다
    payload = {**GOOD, "negatives": ["진짜 반대 1", "반대 근거 확보 실패 [추론]"]}
    v = su.rule_check("stakeholder", "KIVI", payload, {})
    assert v["gap"] == "negatives" and "1건" in v["reason"]


def test_domain_negatives_with_web_source_and_inference_tag_count():
    # 첫 실제 실행 domain:KIVI — "… [웹 URL][추론]" 2건이 0건으로 처리돼 재작업 2회 낭비 (#115)
    negs = ["메모리 제약이 엄격하다[https://www.samsungsds.com/kr/x][추론]",          # 실제 도메인 워커 형식
            "HBM 미지원으로 손실이 크다 [웹 https://v.daum.net/y][추론]"]
    payload = {**eval_(ev("2402.02750", 3, tag="논문") + ev("https://a.com", 1)), "negatives": negs}
    assert su.rule_check("domain", "KIVI", payload, {})["rule"] == "pass"


def test_inference_only_and_duplicate_negatives_not_counted():
    assert su._counts_as_negative("표준화 정보가 부족하다. [추론]") is False          # 추론 단독
    assert su._counts_as_negative("반대 근거 확보 실패 [추론]") is False
    assert su._counts_as_negative("정확도 하락 [논문 p.9]") is True
    assert su._counts_as_negative("플래시 대역폭 한계") is True                       # 태그 없음 — 질은 Judge
    dup = {**GOOD, "negatives": ["같은 문장 [웹 https://a.com]", "같은 문장 [웹 https://a.com]"]}
    assert su.rule_check("stakeholder", "KIVI", dup, {})["gap"] == "negatives"       # 중복은 1건
