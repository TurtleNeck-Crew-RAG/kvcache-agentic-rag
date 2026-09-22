# 파트 분담 · 의존 순서

설계서 7.1 역할표 + 2장 에이전트 표의 "담당" 열을 코드 디렉토리로 옮긴 것.
마감: **개발 산출물 DAY 3 16시** (GitHub 링크 + `RAG-Output_판교_10반_박유진+황재원+민영은+심준용.pdf`, 슬랙 스레드).
발표는 README 로만 10분 — 차별점 + 보고서 핵심 + Lessons Learned.

## 1. 한눈에

| | 이름 | 코드 | 설계서 절 | 산출물 (DoD) |
|---|---|---|---|---|
| **A** | 박유진 | `rag/` 전체 · `agents/tech_research.py` · `experiments/` · `prompts/rag_*.md` `tech_research.md` | 1.3 · 3.4 · 3.5 · 3.7 · 5.4 | ① `python -m rag.indexing` 이 `data/index/` 를 만든다 ② `rag.rag_node.ask()` 가 `[p.N]` 태그 답변 + `RetrievalEntry` 를 돌려준다 ③ `tech_summary[KIVI/InfiniGen]` 5항목 채움 ④ Hit@4 · MRR@4 (sparse 하이브리드) · RAGAS 3종 수치 → README |
| **B** | 심준용 | `agents/market.py` · `agents/stakeholder.py` · `prompts/market.md` `stakeholder.md` · `prompts/rubrics/4.2` `4.3` | 1.1 · 1.2 · 1.4 · 4.2 · 4.3 | ① `market_eval[tech]` Rubric 4.2 로 채움 ② `stakeholder_eval[tech].negatives ≥ 2` (retry 시 "반대 근거 검색" 프롬프트 변형, 실패도 기록) ③ 모든 evidence 에 `[웹 URL]` 태그 + `citations` 누적 |
| **C** | 민영은 | `agents/domain.py` · `agents/synthesis.py` · `prompts/domain.md` `synthesis.md` `neutrality_judge.md` · `prompts/rubrics/4.1` `4.4` `4.5` | 0 · 4.1 · 4.4 · 4.5 · 5.5 | ① `domain_eval[tech]` — 1단계 사실 질의(A 의 `ask()` 사용, "온디바이스"·"적합" 단어 금지) → 2단계 판정 + 3축 ② HW 반례(LLM in a flash · LPDDR-PIM) 웹검색 ③ `synthesis` 매트릭스·conflicts, `trl_estimate` ④ 중립성 Judge → `neutrality` |
| **D** | 황재원 | `graph/` · `agents/report.py` · `app.py` · `prompts/report.md` · `outputs/report/` · README 취합 | 5 · 6 · 취합 | ① `graph/state.py` 확정 (가장 먼저) ② Dispatcher 규칙표 + `tests/test_dispatcher.py` 통과 ③ `python app.py` 끝까지 실행 ④ `report_md` 6장 목차 + REFERENCE 자동 렌더링 → md → PDF ⑤ README · 발표 |

공용: `config/` · `scripts/` · `pyproject.toml` · CI · `README.md` 뼈대 (A 가 초기 세팅, 이후 D 가 취합).

## 2. 의존 그래프 — 누가 누구를 기다리나

```
D  state.py 확정 ──────────────┬──────────────┬──────────────┐
                               │              │              │
A  indexing → retriever →      │              │              │
   judge → rag_node.ask() ─────┼──► C domain  │              │
      │                        │      │       │              │
      ▼                        ▼      │       ▼              │
A  tech_research ─────────► B market  │  B stakeholder       │
      │  (tech_summary)        │      │       │              │
      └────────────────────────┴──────┴───────┘              │
                               │                             │
                               ▼                             │
                        C synthesis + neutrality             │
                               │                             ▼
                               └──────────────────► D report → app.py → PDF
```

| 기다리는 쪽 | 기다리는 것 | 왜 | 막힐 때 우회 |
|---|---|---|---|
| 전원 | **D `state.py`** | 출력 키 이름·타입 | 이미 설계서 5.2 대로 초안이 들어 있음. D 가 1시간 안에 확정 PR |
| B · C | **A `tech_summary`** 스키마 | 평가 워커 입력 | `tests/fixtures/tech_summary.json` 에 손으로 쓴 예시 2건을 두고 그걸로 먼저 개발 |
| C domain | **A `rag.rag_node.ask()`** | 1단계 사실 추출이 RAG | `ask()` 시그니처만 먼저 머지 (A 1순위). C 는 2단계 판정 프롬프트부터 |
| C synthesis | B · C 평가 3개 | 매트릭스 입력 | fixtures 로 먼저. 실제 값은 오후 |
| D report | C `synthesis` · `citations` | 보고서 본문 | 목차·REFERENCE 렌더링은 fixtures 로 먼저. 진짜 데이터는 마지막 2시간 |

**병목은 A 와 D 입니다.** A 의 `ask()` 와 D 의 `state.py` 가 늦으면 전원이 대기하므로 둘은 오전 첫 PR 로 올립니다.

## 3. 권장 순서 (하루 기준)

| 시간 | A 박유진 | B 심준용 | C 민영은 | D 황재원 |
|---|---|---|---|---|
| 오전 1 | 템플릿·뼈대 PR 머지, `fetch_papers` · `indexing` | Rubric 4.2·4.3 → `prompts/rubrics/` | Rubric 4.1·4.4·4.5 → `prompts/rubrics/`, 도메인 1단계 질문 5개 확정 | **`state.py` 확정 PR**, `tests/fixtures/` 예시 State 작성 |
| 오전 2 | retriever(하이브리드) · judge · **`ask()` PR** | `market` 워커 (Tavily, fixtures 입력) | `domain` 2단계 판정 프롬프트, `neutrality_judge` | Dispatcher 테스트 보강, `report` 렌더러(fixtures) |
| 오후 1 | `tech_research` 워커, Hit@4/MRR 실측 (sparse M3 vs BM25) | `stakeholder` 워커 + 반대근거 retry 변형 | `domain` 워커 (진짜 `ask()` 연결), `synthesis` | `app.py` 통합 실행 1회 — 어디서 깨지는지 |
| 오후 2 | RAGAS 3종 측정 → README 수치 | evidence 태그·citations 점검 | 중립성 루프 실제로 도는지, `[추론]` 비율 계산 | 전체 실행 → `report.md` → PDF, README · 발표 |

## 4. 인터페이스 계약 (파트 사이 약속)

- 워커: `run(state: GraphState) -> dict` — **자기 출력 키 + `citations` + `llm_calls`(+RAG 면 `retrieval_log`) 만** 반환
- RAG 공용: `rag.rag_node.ask(tech, question, node) -> {"answer", "evidence", "retrieval_entry", "llm_calls"}` — A 가 만들고 A · C 가 씀
- 프롬프트는 `prompts/*.md` 에서 읽는다. 코드에 긴 문자열 금지
- 기술별 독립 호출: `for tech in ("KIVI", "InfiniGen"): ...` — 한 프롬프트에 둘을 넣지 않음
- evidence 태그: `[논문 p.N]` / `[웹 URL]` / `[추론]` — 태그 없는 판단 문장은 리뷰에서 반려
- 우열·추천 표현 금지 — 중립성 Judge 가 잡지만, 애초에 쓰지 않는다

## 5. 초기 이슈 (템플릿으로 바로 만들 것)

| # | 파트 | 제목 | type |
|---|---|---|---|
| 1 | 공용 | `[CHORE] 리포 뼈대 · 이슈/PR 템플릿 · CI` | chore |
| 2 | D | `[FEAT] State 스키마 확정 + fixtures` | feat |
| 3 | A | `[FEAT] 인덱싱 — PyMuPDF → 600/100 → BGE-M3 캐시 → Chroma + sparse` | feat |
| 4 | A | `[FEAT] rag_node.ask() — 하이브리드 검색 · 관련성 · 재작성 · Faithfulness` | feat |
| 5 | B | `[FEAT] 시장 평가 워커 (Tavily · Rubric 4.2)` | feat |
| 6 | B | `[FEAT] 이해관계자 워커 — 찬반 각 ≥2 · 반대근거 retry` | feat |
| 7 | C | `[FEAT] 도메인 평가 워커 — 사실 추출(RAG) → 판정 · HW 반례 웹검색` | feat |
| 8 | C | `[FEAT] 종합 워커 + 중립성 Judge · TRL` | feat |
| 9 | A | `[FEAT] 기술 조사 워커 — 고정 질문 5 × 기술 2` | feat |
| 10 | D | `[FEAT] 보고서 워커 — 6장 목차 · REFERENCE 렌더링 · md→PDF` | feat |
| 11 | A | `[CHORE] 실측 — sparse M3 vs BM25 Hit@4/MRR · RAGAS` | exp |
| 12 | D | `[CHORE] app.py 통합 실행 · README · 발표` | chore |
