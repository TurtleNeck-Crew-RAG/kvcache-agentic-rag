"""워커 에이전트 — 설계서 2장 표. 인터페이스는 전부 run(state) -> dict.

| 모듈 | 에이전트 | RAG | 도구 | 출력 키 | 소유 |
|---|---|---|---|---|---|
| tech_research | 기술 조사 | O | retriever(tech 필터) | tech_summary | A 박유진 |
| market | 시장 평가 | X | Tavily | market_eval | B 심준용 |
| stakeholder | 이해관계자 평가 | X | Tavily | stakeholder_eval | B 심준용 |
| domain | 도메인 평가 | O | retriever + Tavily | domain_eval | C 민영은 |
| synthesis | 평가 종합 + 중립성 Judge | X | — | synthesis, trl_estimate, neutrality | C 민영은 |
| report | 보고서 생성 | X | — | report_md | D 황재원 |
"""
