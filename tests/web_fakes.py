"""B 파트 워커 테스트용 가짜 검색/LLM. API 호출과 키 없이 실행한다."""
import json
from pathlib import Path
from types import SimpleNamespace

from agents import _web_eval as web

ROOT = Path(__file__).resolve().parents[1]


def state():
    summaries = json.loads((ROOT / "tests/fixtures/tech_summary.json").read_text())
    summaries.pop("_note")
    return {"tech_summary": summaries, "citations": [], "llm_calls": 8, "retry": {}}


def source(tech):
    return {
        "url": f"https://example.org/{tech}", "title": f"{tech} evidence",
        "content": "Open implementation. Community adoption. High memory cost. Complex deployment.",
        "published_date": "2024-06-01",
    }


def claim(tech, text="구현 공개", quote="Open implementation."):
    return {"text": text, "source_url": source(tech)["url"], "quote": quote}


def install(monkeypatch, response):
    queries, prompts = [], []

    def search(**kwargs):
        queries.append(kwargs)
        tech = "KIVI" if '"KIVI"' in kwargs["query"] else "InfiniGen"
        return {"results": [source(tech)]}

    def generate(messages):
        prompts.append(messages)
        return response(json.loads(messages[1][1]))

    monkeypatch.setattr(web, "search_client", lambda: SimpleNamespace(search=search))
    monkeypatch.setattr(web, "generator", lambda schema: SimpleNamespace(invoke=generate))
    # CI가 SDK를 설치하지 않아도 저장소의 실제 프롬프트를 검증한다.
    monkeypatch.setattr(web, "load_prompt", lambda name: (ROOT / f"prompts/{name}.md").read_text())
    return queries, prompts
