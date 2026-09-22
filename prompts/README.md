# prompts/

에이전트별 프롬프트는 코드에 박지 말고 여기 `.md` 로 둔다 (리뷰·diff 가 쉬움). 파일명 = 에이전트 모듈명.

| 파일 | 에이전트 | 소유 | 설계서 |
|---|---|---|---|
| `tech_research.md` | 기술 조사 — 고정 질문 5 (개요·방법·수치·한계·적용조건) | A | 5.4 |
| `market.md` | 시장 평가 | B | 4.2 |
| `stakeholder.md` | 이해관계자 — 찬·반 각 ≥2, retry 시 "반대 근거 검색" 변형 | B | 4.3 |
| `domain.md` | 도메인 — 1단계 사실 질의(“온디바이스”·“적합” 단어 금지) / 2단계 판정 | C | 4.4, 5.5 장치 3 |
| `synthesis.md` | 종합 — 관점×기술 매트릭스, 일치/상충, TRL | C | 4.1, 4.5 |
| `neutrality_judge.md` | 중립성 Judge — 우열·추천 표현 탐지 | C | 5.5 장치 8 |
| `report.md` | 보고서 — 6장 목차 + REFERENCE 표기 형식 | D | 6 |
| `rag_generator.md` | RAG 답변 생성 — `[p.N]` 태그, "문서에 없으면 없다고" | A | 5.4 |
| `rag_relevance.md` / `rag_rewrite.md` / `rag_faithfulness.md` | Judge 1·2, 질문 재작성 | A | 3.7 |
| `rag_translate.md` | 한국어 질의 → BM25 용 영어 검색 질의 (이중 질의 하이브리드) | A | 3.4 실측 |
| `rubrics/` | 4.1~4.5 Rubric 원문 — 프롬프트에서 include | 각 관점 소유자 | 4 |

공통 규칙 (모든 프롬프트에 포함)
- 기술별 독립 호출 — KIVI 와 InfiniGen 을 같은 프롬프트에 넣지 않는다
- 출처 태그 강제 `[논문 p.N]` / `[웹 URL]` / `[추론]`
- 우열·추천 표현 금지 ("더 낫다", "권장", "선택해야")
