"""Supervisor 게이트 — 우선순위 · 상한 · 종료 보장. LLM·네트워크 불필요 (순수 함수 + 가짜 State)."""
import inspect
import json
from pathlib import Path

from langgraph.graph import END

import graph.supervisor as sup
from graph.state import TECHS, init_state
from graph.supervisor import (
    LLM_BUDGET,
    MAX_EVAL,
    MAX_REWORK,
    MAX_STEPS,
    NEXT_NODES,
    WORKERS,
    end_with_warning,
    route,
    supervisor,
)

FIX = Path(__file__).parent / "fixtures"


def _both(v=None):
    return {t: dict(v or {}) for t in TECHS}


def _collected():
    s = init_state({}, {})
    for k in ("tech_summary", "market_eval", "stakeholder_eval", "domain_eval"):
        s[k] = _both({"grade": "중"})
    return s


def _apply(s, out):
    """LangGraph 리듀서 흉내 — step_count · llm_calls 는 add, retry · sufficiency · node_status 는 merge."""
    for k, v in out.items():
        if k in ("step_count", "llm_calls"):
            s[k] = s.get(k, 0) + v
        elif k in ("retry", "sufficiency", "node_status") or k.endswith("_eval") or k == "tech_summary":
            s[k] = {**(s.get(k) or {}), **v}
        else:
            s[k] = v
    return s


def _fail_eval(targets):
    item = {"passed": False, "score": 0.3, "reason": "근거 없는 판단 문장"}
    ok = {"passed": True, "score": 1.0, "reason": ""}
    return {"passed": False, "items": {"groundedness": item, "neutrality": ok, "bias": ok, "coverage": ok},
            "targets": targets, "feedback": "4장 InfiniGen 시장 근거 보강"}


# ── 우선순위 ──

def test_no_llm_in_gate():
    src = inspect.getsource(sup)
    assert "llm(" not in src and "_common" not in src


def test_first_turn_is_tech_research_and_counts_step():
    out = supervisor(init_state({}, {}))
    assert out["next"] == "tech_research" and out["step_count"] == 1 and out["rework_request"] is None


def test_missing_cell_routes_to_its_worker():
    s = _collected()
    s["stakeholder_eval"] = {"KIVI": {}}                          # InfiniGen 미수집
    assert supervisor(s)["next"] == "stakeholder"


def test_insufficient_cell_rework_with_request_and_retry():
    s = _collected()
    s["sufficiency"] = {"market:InfiniGen": {"rule": "fail", "judge": None, "gap": "negatives",
                                             "hint_query": "InfiniGen adoption barrier", "reason": "반대 근거 0건"}}
    out = supervisor(s)
    assert out["next"] == "market"
    assert out["rework_request"] == {"worker": "market", "tech": "InfiniGen", "gap": "negatives",
                                     "hint_query": "InfiniGen adoption barrier"}
    assert out["retry"] == {"market:InfiniGen": 1}


def test_judge_insufficient_also_reworks_but_exhausted_goes_on():
    s = _collected()
    s["sufficiency"] = {"domain:KIVI": {"rule": "pass", "judge": "insufficient", "gap": "", "hint_query": "", "reason": ""}}
    assert supervisor(s)["next"] == "domain"
    s["retry"] = {"domain:KIVI": MAX_REWORK}
    assert supervisor(s)["next"] == "synthesis"                   # 소진 → 다음 단계


def test_budget_stops_rework_only():
    s = _collected()
    s["llm_calls"] = LLM_BUDGET + 1
    s["sufficiency"] = {"market:KIVI": {"rule": "fail", "judge": None, "gap": "", "hint_query": "", "reason": ""}}
    assert supervisor(s)["next"] == "synthesis"


def test_synthesis_then_report_then_end_on_pass():
    s = _collected()
    assert supervisor(s)["next"] == "synthesis"
    s["synthesis"] = {"matrix": {}, "agreements": [], "conflicts": []}
    assert supervisor(s)["next"] == "report"
    s["report_uri"] = "outputs/report/report.md"
    s["eval_result"] = {"passed": True, "items": {}, "targets": [], "feedback": ""}
    out = supervisor(s)
    assert out["next"] == END and out["status"] == "SUCCESS"


def test_eval_fail_worker_target_clears_synthesis_and_report():
    s = _collected()
    s.update(synthesis={"matrix": {}}, report_uri="r.md", eval_result=_fail_eval(["market:InfiniGen"]))
    out = supervisor(s)
    assert out["next"] == "market" and out["eval_attempts"] == 1
    assert out["rework_request"]["tech"] == "InfiniGen" and out["rework_request"]["hint_query"].startswith("4장")
    assert out["synthesis"] is None and out["report_uri"] is None


def test_eval_fail_report_target_keeps_synthesis():
    s = _collected()
    s.update(synthesis={"matrix": {}}, report_uri="r.md", eval_result=_fail_eval(["report"]))
    out = supervisor(s)
    assert out["next"] == "report" and "synthesis" not in out and out["report_uri"] is None


def test_eval_fail_exhausted_ends_with_warning():
    s = _collected()
    s.update(synthesis={"matrix": {}}, report_uri="r.md", eval_result=_fail_eval(["report"]), eval_attempts=MAX_EVAL)
    assert supervisor(s)["next"] == "end_with_warning"


def test_max_steps_first():
    s = init_state({}, {}, max_steps=3)
    s["step_count"] = 3
    assert supervisor(s)["next"] == "end_with_warning"


def test_route_returns_next_and_is_in_edge_targets():
    assert route({"next": "market"}) == "market"
    assert set(NEXT_NODES) == {*WORKERS, "synthesis", "report", "end_with_warning", END}


# ── end_with_warning ──

def test_end_with_warning_appends_unmet_to_report_file(tmp_path):
    md = tmp_path / "report.md"
    md.write_text("# 보고서\n", encoding="utf-8")
    s = _collected()
    s.update(report_uri=str(md), eval_result=_fail_eval(["report"]), eval_attempts=MAX_EVAL,
             sufficiency={"market:InfiniGen": {"rule": "fail", "judge": None, "gap": "", "hint_query": "", "reason": "근거 없음"}},
             retry={"market:InfiniGen": 2})
    out = end_with_warning(s)
    text = md.read_text(encoding="utf-8")
    assert out["status"] == "SUCCESS"
    assert "자동 경고" in text and "market:InfiniGen (재작업 2/2)" in text and "groundedness" in text


def test_end_with_warning_without_report_is_interrupted():
    s = init_state({}, {}, max_steps=1)
    s["step_count"] = 1
    assert end_with_warning(s)["status"] == "INTERRUPTED"


# ── 종료 보장 — 판정이 끝까지 나빠도 상한 안에서 끝난다 ──

def _simulate(always_insufficient: bool, eval_pass: bool):
    s = init_state({}, {})
    visited = []
    for _ in range(200):
        out = supervisor(s)
        _apply(s, out)
        nxt = out["next"]
        visited.append(nxt)
        if nxt == END:
            break
        if nxt == "end_with_warning":
            _apply(s, end_with_warning(s))
            break
        if nxt in WORKERS:                                       # 워커 → assess
            key = sup.PAYLOAD[nxt]
            _apply(s, {key: _both({"grade": "근거 없음"})})
            verdict = {"rule": "fail" if always_insufficient else "pass", "judge": None,
                       "gap": "evidence", "hint_query": "q", "reason": "stub"}
            _apply(s, {"sufficiency": {f"{nxt}:{t}": verdict for t in TECHS}})
        elif nxt == "synthesis":
            s["synthesis"] = {"matrix": {}}
        elif nxt == "report":                                     # report → evaluator
            s["report_md"] = "# r"
            s["eval_result"] = ({"passed": True, "items": {}, "targets": [], "feedback": ""}
                                if eval_pass else _fail_eval(["market:InfiniGen", "report"]))
    return s, visited


def test_happy_path_is_seven_turns():
    s, visited = _simulate(always_insufficient=False, eval_pass=True)
    assert visited == ["tech_research", "market", "stakeholder", "domain", "synthesis", "report", END]
    assert s["step_count"] == 7 and s["status"] == "SUCCESS"


def test_worst_case_terminates_within_limits():
    s, visited = _simulate(always_insufficient=True, eval_pass=False)
    assert visited[-1] == "end_with_warning"
    assert s["step_count"] <= MAX_STEPS
    assert all(n <= MAX_REWORK for n in s["retry"].values())
    assert s["eval_attempts"] == MAX_EVAL
    assert "자동 경고" in s["report_md"]


def test_fixture_states_flow_to_synthesis():
    s = init_state({}, {})
    for name in ("tech_summary.json", "evals.json"):
        d = json.loads((FIX / name).read_text(encoding="utf-8"))
        d.pop("_note", None)
        s.update(d if name == "evals.json" else {"tech_summary": d})
    assert supervisor(s)["next"] == "synthesis"


def test_tech_research_rework_precedes_downstream_collection():
    """tech_summary 는 나머지 워커의 입력 — 기술 조사 셀이 부족하면 시장 수집보다 먼저 보강한다."""
    s = init_state({}, {})
    s["tech_summary"] = _both()
    s["sufficiency"] = {"tech_research:KIVI": {"rule": "fail", "judge": None, "gap": "numbers", "hint_query": "", "reason": ""}}
    assert supervisor(s)["next"] == "tech_research"
    s["retry"] = {"tech_research:KIVI": MAX_REWORK}
    assert supervisor(s)["next"] == "market"
