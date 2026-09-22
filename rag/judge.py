"""Judge — 그림 1(b)·(c).  [소유: A 박유진]

Judge 1  관련성 structured yes/no → no 면 질문 재작성 1회 → 재실패 시 "논문에 근거 없음"
Judge 2  Faithfulness — 생성이 컨텍스트에 근거하는지
모델: llm("judge") = gpt-4.1-mini temperature 0, Generator 와 별도 인스턴스 (설계서 3.6)
"""
from __future__ import annotations

from typing import Literal

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from agents._common import llm, render_prompt


class Relevance(BaseModel):
    relevant: Literal["yes", "no"]
    reason: str = Field(description="한 줄 근거")


class Faithfulness(BaseModel):
    faithful: bool
    unsupported: list[str] = Field(default_factory=list, description="근거 없는 문장들")


def format_context(docs: list[Document]) -> str:
    """각 조각 앞에 [p.N] — Generator 가 그대로 인용하도록."""
    return "\n\n".join(f"[p.{d.metadata['page']}] {d.page_content}" for d in docs)


def check_relevance(question: str, docs: list[Document]) -> bool:
    if not docs:
        return False
    prompt = render_prompt("rag_relevance", question=question, context=format_context(docs))
    out = llm("judge").with_structured_output(Relevance).invoke(prompt)
    return out.relevant == "yes"


def rewrite_query(question: str, tech: str, paper: str, docs: list[Document]) -> str:
    prompt = render_prompt("rag_rewrite", 
        question=question, tech=tech, paper=paper, context=format_context(docs) or "(없음)"
    )
    return llm("judge").invoke(prompt).content.strip().strip('"')


def check_faithfulness(question: str, answer: str, docs: list[Document]) -> Faithfulness:
    prompt = render_prompt("rag_faithfulness", 
        question=question, answer=answer, context=format_context(docs)
    )
    return llm("judge").with_structured_output(Faithfulness).invoke(prompt)
