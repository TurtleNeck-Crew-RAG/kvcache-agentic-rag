"""평가 종합 에이전트

기술·시장·이해관계자·도메인 평가를 관점×기술 매트릭스로 합치고,
일치점·상충점과 공개 정보 기반 TRL 추정을 생성한다.
중립성 검사는 최종 보고서 뒤 evaluator 가 한 곳에서 맡는다 (#83).

출력 키: synthesis · trl_estimate · llm_calls
"""
from __future__ import annotations

import json
import re
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from agents._common import TECHS, llm, load_prompt
from graph.state import GraphState

REQUIRED_INPUTS = ("tech_summary", "market_eval", "stakeholder_eval", "domain_eval")
SOURCE_TAG = re.compile(r"\[(?:논문|웹|추론|p\.\d)[^\]]*\]")   # [논문 p.2, p.9] · [논문 2406.19707 p.9] · [p.3] · [웹 URL] · [추론] — 실출력은 쪽을 여러 개 묶는다(5회차)


def _require_source_tag(value: str) -> str:
    if "근거 없음" not in value and not SOURCE_TAG.search(value):   # "…(논문에 근거 없음)." 도 근거 없음 기록으로 인정
        # 태그 없는 판단 문장 = 근거 없는 추론으로 기록한다 (장치 6). 예외로 워커 전체를 버리면 태그가 있는 나머지 문장까지
        # 사라진다 — 6회차: TRL basis 2문장 때문에 종합 전체가 실패. [추론] 은 한계점 4 의 비율에 그대로 잡힌다.
        return value.rstrip() + " [추론]"
    return value


TaggedText = Annotated[str, AfterValidator(_require_source_tag)]


class TechnologyPairOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    KIVI: TaggedText
    InfiniGen: TaggedText


class MatrixOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    technical_maturity: TechnologyPairOutput = Field(alias="기술 성숙도")
    market: TechnologyPairOutput = Field(alias="시장")
    stakeholder: TechnologyPairOutput = Field(alias="이해관계자")
    domain: TechnologyPairOutput = Field(alias="도메인")


class TRLOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: int = Field(ge=3, le=9)
    basis: list[TaggedText] = Field(min_length=1)
    reference_date: str = Field(min_length=1)


class TRLEstimatesOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    KIVI: TRLOutput
    InfiniGen: TRLOutput


class SynthesisBundleOutput(BaseModel):
    """State.synthesis와 State.trl_estimate를 한 번에 생성하는 스키마."""

    model_config = ConfigDict(extra="forbid")

    matrix: MatrixOutput
    agreements: list[TaggedText] = Field(min_length=1)
    conflicts: list[TaggedText] = Field(min_length=1)
    trl_estimate: TRLEstimatesOutput


def _validate_inputs(state: GraphState) -> None:
    missing = [key for key in REQUIRED_INPUTS if not state.get(key)]
    incomplete: list[str] = []
    for key in REQUIRED_INPUTS:
        values = state.get(key) or {}
        absent_techs = [tech for tech in TECHS if tech not in values]
        if absent_techs:
            incomplete.append(f"{key}({', '.join(absent_techs)})")
    problems = [*missing, *incomplete]
    if problems:
        raise ValueError("synthesis worker requires complete inputs: " + ", ".join(problems))


def _render_prompt(state: GraphState) -> str:
    payload: dict[str, Any] = {
        key: state.get(key)
        for key in ("selected", *REQUIRED_INPUTS)
        if state.get(key) is not None
    }
    parts = [
        load_prompt("synthesis"),
        "## TRL Rubric\n" + load_prompt("rubrics/4.1-trl"),
        "## Synthesis Rubric\n" + load_prompt("rubrics/4.5-synthesis"),
        "## 평가 입력\n" + json.dumps(payload, ensure_ascii=False, indent=2, default=str),
    ]
    return "\n\n".join(parts)


def _unpack(output: SynthesisBundleOutput) -> tuple[dict[str, Any], dict[str, Any]]:
    data = output.model_dump(by_alias=True)
    trl_estimate = data.pop("trl_estimate")
    return data, trl_estimate


def run(state: GraphState) -> dict:
    """관점별 평가를 종합하고 기술별 TRL을 추정한다."""

    _validate_inputs(state)
    evaluator = llm("generator").with_structured_output(SynthesisBundleOutput)
    structured = evaluator.invoke(_render_prompt(state))
    if isinstance(structured, dict):
        structured = SynthesisBundleOutput.model_validate(structured)
    synthesis, trl_estimate = _unpack(structured)
    return {
        "synthesis": synthesis,
        "trl_estimate": trl_estimate,
        "llm_calls": 1,
    }
