"""공용 pytest 설정 — 테스트가 실제 outputs/ 에 쓰지 않게 한다.  [소유: 공용 · D 황재원 작성]

graph/observe.py(A) 의 log_decision · log_retrieval 은 outputs/*.jsonl 에 덧붙인다.
supervisor · 그래프 테스트가 이를 진짜로 부르므로, 테스트마다 출력 위치를 tmp 로 돌린다 (실제 실행 로그와 섞이지 않게).
"""
import pytest


@pytest.fixture(autouse=True)
def _observe_out_to_tmp(tmp_path, monkeypatch):
    try:
        import graph.observe as observe
    except ImportError:                       # observe 머지 전 — supervisor 는 stderr fallback 만 쓴다
        return
    monkeypatch.setattr(observe, "OUT", tmp_path / "outputs")
