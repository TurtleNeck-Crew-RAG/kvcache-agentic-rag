"""웹 검색 횟수 — _web_eval.search 가 질의마다 observe.log_web 을 남기는지 (#102). Tavily 는 가짜."""
import graph.observe as ob
from agents import _web_eval as web


class FakeClient:
    def search(self, query, **kw):
        if "fail" in query:
            raise TimeoutError("down")
        return {"results": [{"url": "https://a.com/x", "content": "c", "title": "t"}] * 2}


def test_search_logs_every_query_including_failures(monkeypatch):
    monkeypatch.setattr(web, "search_client", lambda: FakeClient())
    sources, notes = web.search(["KIVI 시장", "fail query"], node="market", trace_id="W")
    rows = ob.read_jsonl(ob.WEB, "W")
    assert [(r["node"], r["query"], r["n_results"]) for r in rows] == [("market", "KIVI 시장", 2), ("market", "fail query", 0)]
    assert sources and notes                                     # 동작은 그대로 — 실패는 notes 로
