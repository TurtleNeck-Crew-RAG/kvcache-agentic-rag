# 파트 분담 · 시작 가이드

> 이 문서 하나 읽고 바로 개발 시작할 수 있게 썼습니다. 규칙 상세는 [CONTRIBUTING.md](../CONTRIBUTING.md), 설계 근거는 [설계서.md](설계서.md).
>
> **마감: DAY 3 15시** — GitHub 링크 + `RAG-Output_판교_10반_박유진+황재원+민영은+심준용.pdf` (슬랙 스레드).
> 발표는 README 로만 10분 — 차별점 · 보고서 핵심 · Lessons Learned.

---

## 0. 시작 5분 (전원)

```bash
git clone https://github.com/TurtleNeck-Crew-RAG/kvcache-agentic-rag.git
cd kvcache-agentic-rag
uv sync --group dev                 # Python 3.11 .venv (torch 포함이라 몇 분)
cp .env.example .env                # OPENAI_API_KEY · TAVILY_API_KEY · LANGSMITH_API_KEY 채우기
bash scripts/fetch_papers.sh        # data/papers/kivi.pdf · infinigen.pdf (커밋 안 함)
uv run pytest                       # 5개 통과하면 환경 OK
```

그 다음:
1. GitHub 이슈 생성 (아래 **5. 초기 이슈** 표에서 자기 것 골라 템플릿으로) → 번호 확인
2. `git switch -c feat/<번호>-<slug>` (예: `feat/6-stakeholder-worker`)
3. 자기 파일 열어서 `NotImplementedError` 지우고 구현
4. `tests/fixtures/` 로 단독 실행해 보고 → PR

---

## 1. 누가 무엇을

| | 이름 | 한 줄 | 소유 파일 | 설계서 |
|---|---|---|---|---|
| **A** | 박유진 | RAG 파이프라인 + 기술 조사 | `rag/*` · `agents/tech_research.py` · `prompts/tech_research.md` `rag_*.md` · `experiments/` | 1.3 · 3.4 · 3.5 · 3.7 · 5.4 |
| **B** | 심준용 | 시장 · 이해관계자 (웹검색) | `agents/market.py` `stakeholder.py` · `prompts/market.md` `stakeholder.md` · `prompts/rubrics/4.2-market.md` `4.3-stakeholder.md` | 1.1 · 1.2 · 1.4 · 4.2 · 4.3 |
| **C** | 민영은 | 도메인 · 종합 · 중립성 | `agents/domain.py` `synthesis.py` · `prompts/domain.md` `synthesis.md` `neutrality_judge.md` · `prompts/rubrics/4.1-trl.md` `4.4-domain.md` `4.5-synthesis.md` | 0 · 4.1 · 4.4 · 4.5 · 5.5 |
| **D** | 황재원 | State · Graph · 보고서 · 발표 | `graph/*` · `agents/report.py` · `app.py` · `prompts/report.md` · `outputs/report/` · README 취합 | 5 · 6 |

공용 (바꾸기 전에 슬랙): `agents/_common.py` · `config/` · `scripts/` · `pyproject.toml` · CI · `tests/fixtures/`

---

## 2. 파트별 — 오늘 할 일

### A 박유진 — `rag/` · `agents/tech_research.py`

**만들 것**

| 파일 | 함수 | 하는 일 | 설계서 |
|---|---|---|---|
| `rag/indexing.py` | `build_index()` | PyMuPDFLoader → Recursive 600/100 (metadata `chunk_id` · `tech` · `page`) → BGE-M3 + `CacheBackedEmbeddings`(`data/cache/`) → Chroma(`data/index/`) + sparse | 3.4, 그림 1(a) |
| `rag/retriever.py` | `get_retriever(tech)` | Ensemble sparse+dense 0.5/0.5, k=4, `tech` 필터 | 3.4 |
| `rag/judge.py` | `check_relevance` · `rewrite_query` · `check_faithfulness` | Judge 1 (yes/no) · 재작성 · Judge 2 | 3.7, 그림 1(b)(c) |
| `rag/rag_node.py` | **`ask(tech, question, node)`** | 위 셋을 묶은 공용 루틴 — **C 도 이걸 씀** | 5.4 |
| `agents/tech_research.py` | `run(state)` | 고정 질문 5 × `for tech in TECHS` → `ask()` → `tech_summary` | 2장 |
| `rag/evaluate.py` | `main()` | Hit@4 · MRR@4 (20문항) · RAGAS 3종 → README | 3.4 |

**`ask()` 반환 형식** (C 와의 약속)
```python
{
  "answer": "...[p.4]...",                      # 컨텍스트 없으면 "논문에 근거 없음"
  "evidence": [{"claim": "...", "tag": "논문", "ref": "2402.02750", "page": 4}],
  "retrieval_entry": {...},                      # graph.state.RetrievalEntry — 재작성 전/후 둘 다
  "llm_calls": 2,                                # generator 1 + judge 1 (+ rewrite 1)
}
```

**순서**: `indexing` → `retriever` → **`ask()` 시그니처만이라도 먼저 PR** (C 가 기다림) → judge 채우기 → `tech_research` → 실측.
**실측 이슈**: 같은 20문항(`experiments/embed_compare/eval_set.json`)으로 BGE-M3 sparse vs BM25 → Hit@4 ≥ 0.80 인지. 미달이면 chunk 1000 / parent-document 재측정 (3.5 해석 5).

### B 심준용 — `agents/market.py` · `agents/stakeholder.py`

**입력** `state["tech_summary"]` (없으면 `tests/fixtures/tech_summary.json`) · **도구** Tavily · **출력** `market_eval[tech]`, `stakeholder_eval[tech]` (`graph.state.Eval`)

```python
from agents._common import TECHS, llm, load_prompt
from langchain_tavily import TavilySearch

def run(state):
    out, cites, calls = {}, [], 0
    for tech in TECHS:                                    # 기술별 독립 호출
        summary = state["tech_summary"][tech]
        # 1) Tavily 검색 (질의 3~5개: 시장 규모 · 채택 · 생태계 / 이해관계자는 찬·반 각각)
        # 2) llm("generator") + load_prompt("market") + load_prompt("rubrics/4.2-market")
        #    → structured output 으로 Eval 채우기. 모든 evidence 에 [웹 URL] 태그
        out[tech] = ...
        cites += [...]                                    # graph.state.Ref
        calls += 1
    return {"market_eval": out, "citations": cites, "llm_calls": calls}
```

**stakeholder 만의 규칙** (Dispatcher 규칙 2'): `negatives` 가 기술당 **2건 미만이면 재호출**됩니다. `state["retry"].get("stake", 0) > 0` 이면 "반대 근거만 검색" 모드로 프롬프트를 바꾸세요. 2회 후에도 못 채우면 `negatives` 에 `"반대 근거 확보 실패 [추론]"` 을 넣어 기록.
**먼저 할 것**: `prompts/rubrics/4.2-market.md` · `4.3-stakeholder.md` 에 설계서 4.2 · 4.3 표를 옮기기 → 이게 프롬프트의 절반.

### C 민영은 — `agents/domain.py` · `agents/synthesis.py`

**domain** — 2단계 (설계서 4.4)
1. **사실 추출** (RAG): 도메인 질문 5개를 `rag.rag_node.ask(tech, q, node="domain")` 으로. 질문에 **"온디바이스"·"적합" 단어 금지** (장치 3) — 예: "실험에 사용된 메모리 대역폭은?" O / "온디바이스에 적합한가?" X
2. **판정**: 1단계 답 + `config/domain.yaml` 제약 → `DomainEval` (`verdict` 적합/조건부/부적합, `axes` 3축에서 포기한 것). 정확도에 임계값 없음 — 수치는 보고만
3. HW 우호 반례(`domain.yaml` 의 `counter_examples`: LLM in a flash · LPDDR-PIM) 는 **Tavily 로 직접** 검색. 다른 워커 결과는 받지 않음

A 의 `ask()` 가 아직 없으면 → 2단계 판정 프롬프트 + 반례 웹검색부터. `ask()` 자리엔 fixtures 의 `tech_summary` 로 임시.

**synthesis** — 입력 `*_eval` 3개 + `tech_summary` (없으면 `tests/fixtures/evals.json`)
- 관점(TRL · 시장 · 이해관계자 · 도메인) × 기술 매트릭스, `agreements[]`, `conflicts[]` — **conflicts 가 보고서 5장의 본체**
- `trl_estimate[tech]` = `{level, basis[], reference_date}` (4.1 Rubric, 기준 시점 명시)
- **중립성 Judge**: `llm("judge")` + `prompts/neutrality_judge.md` 로 우열·추천 표현 탐지 → `neutrality = {"result": "pass"|"fail", "violations": [...]}`. fail 이면 Dispatcher 가 재호출 (≤2). `retry["synth"] > 0` 이면 violations 를 프롬프트에 넣어 고치게
- 반환: `{"synthesis", "trl_estimate", "neutrality", "llm_calls"}`

**먼저 할 것**: `prompts/rubrics/4.1-trl.md` `4.4-domain.md` `4.5-synthesis.md` 옮기기, 도메인 1단계 질문 5개 확정.

### D 황재원 — `graph/` · `agents/report.py` · `app.py`

**이미 있음**: `graph/state.py`(5.2 표 그대로) · `graph/dispatcher.py`(5.3 규칙표) · `graph/build.py` · `tests/test_dispatcher.py` 5개 통과.
**첫 PR (오전 1시간 안)**: `state.py` 를 훑고 확정 — 키 이름·타입에 이견 있으면 지금 바꾸고 슬랙 공지. 이후 변경은 이슈로.

**report** — 입력 `synthesis` · `trl_estimate` · `*_eval` · `tech_summary` · `citations` · `retrieval_log` (없으면 fixtures)
- 설계서 6장 목차 그대로: SUMMARY(½p) → 1 배경 → 2 선정(`config/selection.yaml`) → 3 개요(`tech_summary`) → 4 관점별(각 관점 안에서 KIVI · InfiniGen 나란히) → 5 시사점(`conflicts`) → 6 한계점(6항목 — `[추론]` 비율 · Hit@4 · 재작성 효과는 `retrieval_log` 에서 계산) → REFERENCE
- REFERENCE: `citations` 중 **본문에 인용된 것만**, 6장 표기 형식으로 렌더링 (논문/특허/웹)
- 장별로 LLM 호출을 나누면 (SUMMARY 는 마지막) 컨텍스트가 안 터짐. `neutrality.violations` 가 남아 있으면 6장에 표시
- md → PDF: `uv sync --extra pdf` (weasyprint, `brew install pango`) 안 되면 VS Code Markdown PDF 로 수동. 파일명 `RAG-Output_판교_10반_박유진+황재원+민영은+심준용.pdf`

**app.py**: 이미 뼈대 있음. 인덱스 없으면 `build_index()` 호출, LangSmith 프로젝트 `kv-cache-eval` 확인, 오후 첫 통합 실행에서 어디서 깨지는지 잡는 게 D 의 일.

---

## 3. 의존 그래프 — 누가 누구를 기다리나

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

| 기다리는 쪽 | 기다리는 것 | 막힐 때 우회 |
|---|---|---|
| 전원 | D `state.py` | 이미 설계서대로 들어 있음 — 확정 PR 전에도 그 키 이름으로 개발 시작 |
| B · C | A `tech_summary` | `tests/fixtures/tech_summary.json` |
| C domain | A `ask()` | 시그니처 먼저 머지. C 는 2단계 판정 + 반례 웹검색부터 |
| C synthesis | 평가 3개 | `tests/fixtures/evals.json` |
| D report | C `synthesis` · `citations` | fixtures 로 렌더링부터. 진짜 데이터는 마지막 2시간 |

**병목은 A 의 `ask()` 와 D 의 `state.py`.** 둘은 오전 첫 PR.

---

## 4. 인터페이스 계약 (파트 사이 약속)

| 약속 | 내용 |
|---|---|
| 워커 시그니처 | `run(state: GraphState) -> dict` — **자기 출력 키 + `citations` + `llm_calls`** (+RAG 면 `retrieval_log`) 만 반환. 다른 키 건드리지 않음 |
| 기술별 독립 | `for tech in TECHS:` — 한 프롬프트에 KIVI 와 InfiniGen 을 같이 넣지 않음 (장치 2) |
| 프롬프트 | `prompts/<worker>.md` 에 두고 `load_prompt()` 로 읽음. Rubric 은 `load_prompt("rubrics/4.x-...")` 로 include |
| LLM | `llm("generator")` 생성 · `llm("judge")` 판정(temperature 0) · `llm("light")` 형식 변환만. 직접 `ChatOpenAI(...)` 만들지 않음 |
| 출처 태그 | 모든 판단 문장에 `[논문 p.N]` / `[웹 URL]` / `[추론]`. `evidence[]` 에도 `tag` 필드 |
| citations | `graph.state.Ref` 형식 `{type, authors, year, title, venue, id_or_url, accessed}` — 보고서가 6장 형식으로 렌더링 |
| 금지 표현 | "더 낫다" · "권장" · "선택해야" · "우수" — 중립성 Judge 가 잡지만 애초에 쓰지 않음 |
| llm_calls | LLM 을 부른 횟수를 정직하게 더함 — `> 100` 이면 Dispatcher 가 재호출을 끔 |
| structured output | Eval · DomainEval · TechSummary 는 Pydantic 모델로 `with_structured_output` 권장 — 키 누락 방지 |

---

## 5. 초기 이슈 — 템플릿으로 바로 만들 것

| # | 파트 | 제목 | type |
|---|---|---|---|
| 1 | 공용 | `[CHORE] 리포 뼈대 · 이슈/PR 템플릿 · CI` (완료) | chore |
| 2 | D | `[FEAT] State 스키마 확정 + fixtures 검토` | feat |
| 3 | A | `[FEAT] 인덱싱 — PyMuPDF → 600/100 → BGE-M3 캐시 → Chroma + sparse` | feat |
| 4 | A | `[FEAT] rag_node.ask() — 하이브리드 검색 · 관련성 · 재작성 · Faithfulness` | feat |
| 5 | B | `[FEAT] 시장 평가 워커 (Tavily · Rubric 4.2)` | feat |
| 6 | B | `[FEAT] 이해관계자 워커 — 찬반 각 ≥2 · 반대근거 retry` | feat |
| 7 | C | `[FEAT] 도메인 평가 워커 — 사실 추출(RAG) → 판정 · HW 반례 웹검색` | feat |
| 8 | C | `[FEAT] 종합 워커 + 중립성 Judge · TRL` | feat |
| 9 | A | `[FEAT] 기술 조사 워커 — 고정 질문 5 × 기술 2` | feat |
| 10 | D | `[FEAT] 보고서 워커 — 6장 목차 · REFERENCE 렌더링 · md→PDF` | feat |
| 11 | A | `[CHORE] 실측 — sparse M3 vs BM25 Hit@4/MRR · RAGAS` | exp |
| 12 | D | `[CHORE] app.py 통합 실행 · README 수치 · 발표` | chore |

---

## 6. 시간표 (하루 기준)

| | A 박유진 | B 심준용 | C 민영은 | D 황재원 |
|---|---|---|---|---|
| 오전 1 | `indexing` · `retriever` | Rubric 4.2 · 4.3 → `prompts/rubrics/` | Rubric 4.1 · 4.4 · 4.5, 도메인 질문 5개 확정 | **`state.py` 확정 PR**, fixtures 검토 |
| 오전 2 | judge · **`ask()` PR** | `market` 워커 (fixtures 입력) | `domain` 2단계 판정 · `neutrality_judge` | `report` 렌더러 (fixtures) |
| 오후 1 | `tech_research`, sparse M3 vs BM25 실측 | `stakeholder` + 반대근거 retry | `domain` 에 진짜 `ask()` 연결 · `synthesis` | `app.py` 통합 실행 1회 — 어디서 깨지나 |
| 오후 2 | RAGAS → README 수치 | evidence 태그 · citations 점검 | 중립성 루프 실동작 · `[추론]` 비율 | 전체 실행 → `report.md` → PDF · README · 발표 |

---

## 7. 자주 물을 것

- **Q. 임베딩 모델 다운로드가 느려요** — BGE-M3 2.27GB, 첫 `rag.indexing` 때 한 번. 그 전엔 B · C · D 는 fixtures 로 개발하면 됨.
- **Q. 워커 하나만 돌려보고 싶어요** — `tests/fixtures/README.md` 의 스니펫. `app.py` 전체는 D 가 오후에.
- **Q. State 에 키 하나 추가하고 싶어요** — `[CHORE] 설계 변경` 이슈 → D. 워커 안에서 필요한 중간값은 반환하지 말고 로컬 변수로.
- **Q. 논문 밖 자료를 인덱싱해도 되나요** — 안 됩니다 (과제 명세 "제시된 풀"). 반례 · 시장 자료는 Tavily.
- **Q. 설계서와 다르게 만들었어요** — PR "설계서 반영" 칸에 절 번호 + 보고서 6장 한계점 또는 README Lessons Learned 에 한 줄. 설명 없으면 "설계 구현 충실도" 감점.
- **Q. 이슈에 설계서 근거 꼭 써야 하나요** — 필수 아님. 있으면 리뷰가 빨라짐.
