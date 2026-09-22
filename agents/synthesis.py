"""평가 종합 에이전트 — 설계서 2장, 4.1·4.5 Rubric.  [소유: C 민영은]

기술·시장·이해관계자·도메인 평가를 관점×기술 매트릭스로 합치고,
일치점·상충점과 공개 정보 기반 TRL 추정을 생성한다. 중립성 Judge와
재작성 루프는 #15에서 결합한다.

출력 키: synthesis · trl_estimate · llm_calls
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agents._common import TECHS, llm, load_prompt
from graph.state import GraphState

REQUIRED_INPUTS = ("tech_summary", "market_eval", "stakeholder_eval", "domain_eval")


class TechnologyPairOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    KIVI: str = Field(min_length=1)
    InfiniGen: str = Field(min_length=1)


class MatrixOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    technical_maturity: TechnologyPairOutput = Field(alias="기술 성숙도")
    market: TechnologyPairOutput = Field(alias="시장")
    stakeholder: TechnologyPairOutput = Field(alias="이해관계자")
    domain: TechnologyPairOutput = Field(alias="도메인")


class TRLOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: int = Field(ge=3, le=9)
    basis: list[str] = Field(min_length=1)
    reference_date: str = Field(min_length=1)


class TRLEstimatesOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    KIVI: TRLOutput
    InfiniGen: TRLOutput


class SynthesisBundleOutput(BaseModel):
    """State.synthesis와 State.trl_estimate를 한 번에 생성하는 스키마."""

    model_config = ConfigDict(extra="forbid")

    matrix: MatrixOutput
    agreements: list[str] = Field(min_length=1)
    conflicts: list[str] = Field(min_length=1)
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
    return "\n\n".join(
        (
            load_prompt("synthesis"),
            "## TRL Rubric\n" + load_prompt("rubrics/4.1-trl"),
            "## Synthesis Rubric\n" + load_prompt("rubrics/4.5-synthesis"),
            "## 평가 입력\n"
            + json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        )
    )


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
