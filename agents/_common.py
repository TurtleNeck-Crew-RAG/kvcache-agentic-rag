"""워커 공용 유틸 — 프롬프트 로딩 · LLM 인스턴스 · 기술 목록.  [공용 — 바꾸기 전에 슬랙]

from agents._common import TECHS, llm, load_prompt, render_prompt

    prompt = render_prompt("market", tech=tech)    # prompts/market.md 의 {tech} 치환
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
# 호출 1회 상한 (#102): 지정하지 않으면 OpenAI 클라이언트 기본값(600초)이라 호출 하나가 멈추면 실행 전체가 기록 없이 선다.
# 실행 전체의 벽시계 상한(app.py RUN_TIMEOUT)은 노드 경계에서만 보므로, 노드 안의 멈춤은 여기서 끊는다.
CALL_LIMITS = dict(timeout=60, max_retries=2)
MODELS = {
    "generator": dict(model="gpt-4.1-mini", temperature=0.2, **CALL_LIMITS),
    "judge": dict(model="gpt-4.1-mini", temperature=0, **CALL_LIMITS),
    "light": dict(model="gpt-4.1-nano", temperature=0, **CALL_LIMITS),   # 인용 형식 정리 등 판단 없는 변환만
}


def load_prompt(name: str) -> str:
    """prompts/<name>.md 를 읽는다. 코드에 긴 프롬프트 문자열을 두지 않기 위함."""
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


def render_prompt(name: str, **values: object) -> str:
    """load_prompt + `{key}` 치환. str.format 과 달리 프롬프트 안의 다른 중괄호(JSON 예시 등)를 건드리지 않는다."""
    text = load_prompt(name)
    for key, val in values.items():
        text = text.replace("{" + key + "}", str(val))
    return text


@lru_cache
def llm(role: Literal["generator", "judge", "light"]) -> ChatOpenAI:
    return ChatOpenAI(**MODELS[role])
