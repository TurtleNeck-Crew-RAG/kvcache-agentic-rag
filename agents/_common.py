"""워커 공용 유틸 — 프롬프트 로딩 · LLM 인스턴스 · 기술 목록.  [공용 — 바꾸기 전에 슬랙]

from agents._common import TECHS, load_prompt, llm

    prompt = load_prompt("market")                 # prompts/market.md
    rubric = load_prompt("rubrics/4.2-market")     # prompts/rubrics/4.2-market.md
    out = llm("generator").invoke(...)             # gpt-4.1-mini
    out = llm("judge").invoke(...)                 # gpt-4.1-mini, temperature 0, 별도 인스턴스
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from langchain_openai import ChatOpenAI

TECHS = ("KIVI", "InfiniGen")          # 기술별 독립 호출 — for tech in TECHS 로 돈다
PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

# 설계서 3.6
MODELS = {
    "generator": dict(model="gpt-4.1-mini", temperature=0.2),
    "judge": dict(model="gpt-4.1-mini", temperature=0),
    "light": dict(model="gpt-4.1-nano", temperature=0),   # 인용 형식 정리 등 판단 없는 변환만
}


def load_prompt(name: str) -> str:
    """prompts/<name>.md 를 읽는다. 코드에 긴 프롬프트 문자열을 두지 않기 위함."""
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


@lru_cache
def llm(role: Literal["generator", "judge", "light"]) -> ChatOpenAI:
    return ChatOpenAI(**MODELS[role])
