"""시장·이해관계자 전용 웹 근거 처리. [소유: B 심준용]

State에는 검색 원문을 넣지 않는다. 인용 URL과 발췌문은 실제 검색 결과와 대조한다.
SDK는 호출 시 import하여 네트워크 없는 CI에서도 순수 변환 로직을 검증한다.
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Literal, get_args
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from graph.state import Eval, Ref, Tech

# State의 확정된 기술 집합. SDK 없이도 워커를 로드할 수 있다.
TECHS = get_args(Tech)


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, description="한국어 판단 한 문장. 출처 태그는 코드가 추가한다.")
    source_url: str | None = Field(description="제공된 검색 결과 URL. 추론이면 null.")
    quote: str = Field(description="검색 content에 실제 있는 연속 발췌문. 추론이면 빈 문자열.")


class MarketAxis(BaseModel):
    grade: Literal["상", "중", "하", "근거 없음"]
    reason: Claim


def load_prompt(name: str) -> str:
    from agents._common import load_prompt as common_load_prompt

    return common_load_prompt(name)


def search_client():
    from tavily import TavilyClient

    return TavilyClient()


def generator(schema):
    from agents._common import llm

    return llm("generator").model_copy(update={"max_retries": 0}).with_structured_output(schema)


def search(queries: list[str]) -> tuple[dict[str, dict], list[str]]:
    sources, notes = {}, []
    try:
        client = search_client()
    except Exception as exc:
        return {}, [f"웹 검색 초기화 실패 ({type(exc).__name__})"]
    for query in queries:
        try:
            response = client.search(
                query=query, search_depth="advanced", max_results=5,
                include_answer=False, include_raw_content=False, timeout=20,
            )
            results = response["results"]
            if not isinstance(results, list):
                raise ValueError("invalid search results")
            for item in results:
                if not isinstance(item, dict):
                    continue
                url, content = item.get("url"), item.get("content")
                if not isinstance(url, str) or not isinstance(content, str) or not content.strip():
                    continue
                parsed = urlsplit(url)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    continue
                snippet = content[:3000]
                if url in sources:
                    if snippet not in sources[url]["content"]:
                        sources[url]["content"] = (sources[url]["content"] + "\n" + snippet)[:6000]
                else:
                    sources[url] = {
                        "url": url, "title": str(item.get("title") or parsed.netloc),
                        "content": snippet, "published_date": str(item.get("published_date") or ""),
                    }
        except Exception as exc:
            # 예외 메시지에는 요청 헤더/API 키가 들어갈 수 있어 타입만 기록한다.
            notes.append(f"웹 검색 일부 실패 ({type(exc).__name__})")
    return sources, list(dict.fromkeys(notes))


def messages(worker: str, rubric: str, tech: str, summary: dict, sources: dict, **extra) -> list:
    payload = {
        "tech": tech, "reference_date": date.today().isoformat(),
        "tech_summary": summary, "sources": list(sources.values()), **extra,
    }
    return [
        ("system", load_prompt(worker) + "\n\n" + load_prompt(rubric)),
        ("human", json.dumps(payload, ensure_ascii=False)),
    ]


def blank(reason: str) -> Eval:
    result = Eval(grade="평가 불가", rationale="", positives=[], negatives=[], evidence=[], confidence=0.0)
    add_note(result, reason)
    return result


def add_note(result: Eval, note: str) -> None:
    text = f"{note} [추론]"
    if text not in result["rationale"]:
        result["rationale"] = (result["rationale"] + "\n" + text).strip()
        result["evidence"].append({"claim": text, "tag": "추론", "ref": "", "page": None})


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


class Grounding:
    """생성 모델이 만든 URL/발췌문을 검증하고 사용한 citation만 만든다."""

    def __init__(self, sources: dict):
        self.sources = sources
        self.evidence = []
        self.used = set()
        self.rejected = 0

    def claim(self, claim: Claim, *, require_web: bool = False) -> str | None:
        if claim.source_url is None:
            if require_web:
                self.rejected += 1
                return None
            text, tag, ref = f"{claim.text} [추론]", "추론", ""
        else:
            source = self.sources.get(claim.source_url)
            quote = _normalized(claim.quote)
            if not source or not quote or quote not in _normalized(source["content"]):
                self.rejected += 1
                return None
            text, tag, ref = f"{claim.text} [웹 {claim.source_url}]", "웹", claim.source_url
            self.used.add(ref)
        entry = {"claim": text, "tag": tag, "ref": ref, "page": None}
        if entry not in self.evidence:
            self.evidence.append(entry)
        return text

    def claims(self, claims: list[Claim], *, require_web: bool = False) -> list[str]:
        output, seen = [], set()
        for claim in claims:
            # 같은 발췌문을 바꿔 말한 두 문장으로 최소 건수를 채우지 않는다.
            key = (claim.source_url, _normalized(claim.quote or claim.text))
            if key in seen:
                continue
            text = self.claim(claim, require_web=require_web)
            if text:
                seen.add(key)
                output.append(text)
        return output

    def citations(self) -> list[Ref]:
        citations = []
        for url, source in self.sources.items():
            if url not in self.used:
                continue
            year = re.match(r"(\d{4})\b", source["published_date"])
            citations.append(Ref(
                type="웹", authors="저자 미상", year=year.group(1) if year else "연도 미상",
                title=source["title"], venue=urlsplit(url).netloc, id_or_url=url,
                accessed=date.today().isoformat(),
            ))
        return citations


def new_citations(citations: list[Ref], existing: list[Ref]) -> list[Ref]:
    seen = {ref["id_or_url"] for ref in existing}
    output = []
    for ref in citations:
        if ref["id_or_url"] not in seen:
            seen.add(ref["id_or_url"])
            output.append(ref)
    return output
