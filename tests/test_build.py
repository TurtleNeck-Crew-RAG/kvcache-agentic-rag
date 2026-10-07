"""graph/build.py 통합 — fixtures stub 으로 START → END (LLM·네트워크 없음). 체크포인터 · 재개 포함."""
from langgraph.checkpoint.memory import InMemorySaver

from graph.build import build_graph
from graph.state import init_state
from graph.supervisor import MAX_EVAL, MAX_REWORK, RECURSION_LIMIT
from tests.fixtures.stubs import make_assess, make_evaluator, make_workers


def _run(calls, **kw):
    workers = make_workers(calls, fail=kw.pop("fail", set()))
    app = build_graph(workers=workers, assess=kw.pop("assess", make_assess()),
                      evaluator=kw.pop("evaluator", make_evaluator()), checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t-test"}, "recursion_limit": RECURSION_LIMIT}
    out = app.invoke(init_state({}, {}, trace_id="t-test"), cfg)
    return out, app, cfg


def test_happy_path_reaches_end():
    calls = []
    out, _, _ = _run(calls)
    assert calls == ["tech_research", "market", "stakeholder", "domain", "synthesis", "report"]
    assert out["status"] == "SUCCESS" and out["step_count"] == 7 and out["eval_result"]["passed"]
    assert out["trace_id"] == "t-test" and set(out["node_status"].values()) == {"ok"}


def test_rework_and_eval_loop_show_in_trace():
    calls = []
    out, _, _ = _run(calls, assess=make_assess({"market:InfiniGen": 1}),
                     evaluator=make_evaluator(fail_times=1, targets=["stakeholder:KIVI"]))
    assert "market:InfiniGen" in calls                       # 셀 재작업 — 그 기술만
    assert "stakeholder:KIVI" in calls                       # 평가 fail → 해당 관점 재조사
    assert calls.count("synthesis") == 2 and calls.count("report") == 2
    assert out["retry"] == {"market:InfiniGen": 1, "stakeholder:KIVI": 1} and out["eval_attempts"] == 1
    assert set(out["market_eval"]) == {"KIVI", "InfiniGen"}  # merge_by_tech — 재작업이 KIVI 를 지우지 않는다
    assert out["status"] == "SUCCESS"


def test_eval_never_passes_ends_with_warning():
    calls = []
    out, _, _ = _run(calls, evaluator=make_evaluator(fail_times=99))
    assert out["eval_attempts"] == MAX_EVAL and calls.count("report") == MAX_EVAL + 1
    assert "자동 경고" in out["report_md"]


def test_failed_worker_is_marked_reworked_and_not_success():
    calls = []
    out, _, _ = _run(calls, fail={"market"})
    assert out["node_status"]["market"] == "failed" and out["last_error"]["node"] == "market"
    assert calls.count("market") == 1 and calls.count("market:KIVI") == MAX_REWORK   # 실패 셀 → 재작업 상한까지
    assert out["retry"]["market:InfiniGen"] == MAX_REWORK
    assert len(out["errors"]) == 1 + 2 * MAX_REWORK
    assert "자동 경고" in out["report_md"] and "노드 실패 market" in out["report_md"]


def test_pending_judges_still_reach_end():
    """A assess · C evaluator 머지 전 — 판정 없이도 END (점심 직후 첫 통합 실행 조건)."""
    calls = []
    workers = make_workers(calls)
    out = build_graph(workers=workers).invoke(init_state({}, {}), {"recursion_limit": RECURSION_LIMIT})
    assert calls[-1] == "report" and out["status"] == "SUCCESS"


def test_resume_from_checkpoint_after_crash():
    calls = []
    workers = make_workers(calls)
    saver = InMemorySaver()
    app = build_graph(workers=workers, assess=make_assess(), evaluator=make_evaluator(), checkpointer=saver)
    cfg = {"configurable": {"thread_id": "t-resume"}, "recursion_limit": RECURSION_LIMIT}
    for i, _ in enumerate(app.stream(init_state({}, {}, trace_id="t-resume"), cfg, stream_mode="updates")):
        if i == 4:                                            # supervisor · tech_research · assess · supervisor · market 까지 하고 "죽는다"
            break
    assert calls == ["tech_research", "market"]
    out = app.invoke(None, cfg)                               # --resume — 성공한 노드는 다시 돌지 않는다
    assert calls == ["tech_research", "market", "stakeholder", "domain", "synthesis", "report"]
    assert out["status"] == "SUCCESS"


def test_all_workers_fail_still_terminates():
    """워커가 전부 예외를 내도 상한 안에서 끝난다 — RAG 때 test_graph_reaches_end_when_all_workers_fail 의 그래프 판."""
    calls = []
    out, _, _ = _run(calls, fail={"tech_research", "market", "stakeholder", "domain"})
    assert out["next"] == "end_with_warning" and out["step_count"] <= out["max_steps"]
    assert all(n == MAX_REWORK for n in out["retry"].values()) and len(out["retry"]) == 8
    assert "자동 경고" in out["report_md"] and "노드 실패 domain" in out["report_md"]


def test_sqlite_checkpoint_resumes_in_new_graph(tmp_path):
    """app.py --resume — 프로세스가 죽고 새 그래프 인스턴스가 같은 sqlite · thread_id 로 이어 간다 (#91)."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    db = tmp_path / "checkpoints.sqlite"
    cfg = {"configurable": {"thread_id": "t-sqlite"}, "recursion_limit": RECURSION_LIMIT}

    def app(calls):
        saver = SqliteSaver(sqlite3.connect(db, check_same_thread=False))
        return build_graph(workers=make_workers(calls), assess=make_assess(), evaluator=make_evaluator(), checkpointer=saver)

    first = []
    for i, _ in enumerate(app(first).stream(init_state({}, {}, trace_id="t-sqlite"), cfg, stream_mode="updates")):
        if i == 4:
            break
    second = []
    out = app(second).invoke(None, cfg)                       # 메모리를 공유하지 않는 새 인스턴스
    assert first == ["tech_research", "market"]
    assert second == ["stakeholder", "domain", "synthesis", "report"] and out["status"] == "SUCCESS"
