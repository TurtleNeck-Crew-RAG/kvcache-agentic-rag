import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import agents.synthesis as synthesis

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    data.pop("_note", None)
    return data


def _state() -> dict:
    state = {"tech_summary": _load("tech_summary.json")}
    state.update(_load("evals.json"))
    state["selected"] = {
        "sw": {"name": "KIVI", "venue": "ICML 2024"},
        "hw": {"name": "InfiniGen", "venue": "OSDI 2024"},
    }
    return state


def _bundle() -> synthesis.SynthesisBundleOutput:
    return synthesis.SynthesisBundleOutput(
        matrix={
            "기술 성숙도": {
                "KIVI": "TRL 4 추정 — 공개 코드와 후속 재현 근거 [웹 URL]",
                "InfiniGen": "TRL 3 추정 — 학회 발표 근거 [논문 p.1]",
            },
            "시장": {
                "KIVI": "오픈소스 구현은 있으나 공식 통합은 미확인 [웹 URL]",
                "InfiniGen": "학술 구현 단계로 제품 적용은 미확인 [추론]",
            },
            "이해관계자": {
                "KIVI": "무보정 적용과 프레임워크 통합 장벽이 함께 보고됨 [웹 URL]",
                "InfiniGen": "무손실 접근과 모바일 검증 부족이 함께 보고됨 [웹 URL]",
            },
            "도메인": {
                "KIVI": "재학습은 불필요하지만 정확도 손실은 별도 보고 [논문 p.1]",
                "InfiniGen": "메모리를 확보하지만 전송 지연은 모바일 미검증 [추론]",
            },
        },
        agreements=["두 기술 모두 배포 환경의 메모리 제약을 다룬다 [논문 p.1]"],
        conflicts=["메모리 절감의 효과와 품질·지연 비용을 관점별로 다르게 해석한다 [추론]"],
        trl_estimate={
            "KIVI": {
                "level": 4,
                "basis": ["공개 코드와 후속 재현 근거 [웹 URL]"],
                "reference_date": "2024 공개 자료 기준",
            },
            "InfiniGen": {
                "level": 3,
                "basis": ["OSDI 2024 게재 [논문 p.1]"],
                "reference_date": "2024 공개 자료 기준",
            },
        },
    )


def test_run_builds_matrix_and_trl_from_all_evaluations(monkeypatch):
    synthesis_prompts: list[str] = []
    roles: list[str] = []

    class FakeEvaluator:
        def invoke(self, prompt: str):
            synthesis_prompts.append(prompt)
            return _bundle()

    generator = Mock()
    generator.with_structured_output.return_value = FakeEvaluator()

    def fake_llm(role):
        roles.append(role)
        return generator
    monkeypatch.setattr(synthesis, "llm", fake_llm)

    result = synthesis.run(_state())

    assert set(result) == {"synthesis", "trl_estimate", "llm_calls"}
    assert result["llm_calls"] == 1 and roles == ["generator"]   # 중립성 Judge 는 evaluator 로 이전 (#83)
    assert set(result["synthesis"]["matrix"]) == {
        "기술 성숙도",
        "시장",
        "이해관계자",
        "도메인",
    }
    assert set(result["trl_estimate"]) == {"KIVI", "InfiniGen"}
    assert result["trl_estimate"]["KIVI"]["level"] == 4
    assert len(result["synthesis"]["conflicts"]) >= 1

    assert len(synthesis_prompts) == 1
    for key in synthesis.REQUIRED_INPUTS:
        assert f'"{key}"' in synthesis_prompts[0]


def test_run_validates_dict_structured_output(monkeypatch):
    class FakeEvaluator:
        def invoke(self, prompt: str):
            return _bundle().model_dump(by_alias=True)

    generator = Mock()
    generator.with_structured_output.return_value = FakeEvaluator()
    monkeypatch.setattr(synthesis, "llm", lambda role: generator)

    result = synthesis.run(_state())

    assert result["synthesis"]["matrix"]["도메인"]["InfiniGen"]
    assert "neutrality" not in result


def test_run_ignores_legacy_neutrality_retry(monkeypatch):
    """이행 기간에 State 에 남은 neutrality · retry["synth"] 가 있어도 재작성 요청을 붙이지 않는다."""
    prompts: list[str] = []

    class FakeEvaluator:
        def invoke(self, prompt: str):
            prompts.append(prompt)
            return _bundle()

    generator = Mock()
    generator.with_structured_output.return_value = FakeEvaluator()
    monkeypatch.setattr(synthesis, "llm", lambda role: generator)
    state = _state() | {"retry": {"synth": 1}, "neutrality": {"result": "fail", "violations": ["KIVI가 더 우수하다"]}}

    synthesis.run(state)

    assert "재작성 요청" not in prompts[0] and "KIVI가 더 우수하다" not in prompts[0]


@pytest.mark.parametrize("missing", synthesis.REQUIRED_INPUTS)
def test_run_rejects_missing_required_input(monkeypatch, missing):
    state = _state()
    state.pop(missing)

    with pytest.raises(ValueError, match=missing):
        synthesis.run(state)


def test_run_rejects_missing_technology():
    state = _state()
    state["market_eval"].pop("InfiniGen")

    with pytest.raises(ValueError, match=r"market_eval\(InfiniGen\)"):
        synthesis.run(state)
