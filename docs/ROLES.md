# 파트 분담 · 시작 가이드 — Agent 과제 (Multi-Agent Orchestration)

> 브랜치 `agent-supervisor` 기준 문서입니다. RAG 과제 분담은 `main` 의 같은 파일에 그대로 있습니다.
> 규칙 상세는 [CONTRIBUTING.md](../CONTRIBUTING.md), 명세 정본은 강사 노션 「Multi-Agent Orchestration」(충돌하면 노션이 이김).
>
> **마감: DAY 2 (2026-10-07) 퇴근 전** — Slack 반별 스레드에 Git 링크 + `tracing-N.png` + 보고서 PDF,
> 파일은 `Agent_판교_10반_박유진+황재원+민영은+심준용.zip`

---

## 0. 무엇을 바꾸나 — 한 장 요약

RAG 과제의 Dispatcher 는 허브 구조였지만 **규칙 1 → 2 → 3 → 4 순서가 코드에 고정**돼 있었다.
노션 필수 항목("순서 하드코딩 금지 · 스텝 수 고정 금지 · 근거 충분성 평가 후 보고서 · 부족하면 해당 하위에 재작업")을
맞추려면 **Supervisor 가 State 의 부족분을 보고 매 턴 다음 담당 1명을 고르는** 구조로 바꿔야 한다.

### 패턴: Supervisor (Hybrid 판정 + 결정론 게이트)

```
START → supervisor ──route()──┬─ tech_research ─┐
          ▲                   ├─ market ────────┤
          │                   ├─ stakeholder ───┼─→ assess ──┐   (충분성: 규칙 → LLM Judge)
          │                   ├─ domain ────────┘            │
          │                   ├─ synthesis ──────────────────┤
          │                   ├─ report ─→ evaluator ────────┤   (품질: 규칙 → LLM Judge)
          │                   ├─ end_with_warning → END      │
          │                   └─ END                         │
          └──────────────────────────────────────────────────┘
```

| 층 | 노드 | 하는 일 | 성격 |
|---|---|---|---|
| 판정 (Judge) | `assess` · `evaluator` | **규칙 검사가 먼저 거르고, 통과한 것만 LLM Judge** — 결과를 State 에 기록만 한다 | 확률 (LLM) |
| 게이트 (Gate) | `supervisor` + `route()` | State 의 판정 결과 · 시도 횟수 · 상한만 보고 `next` 를 정한다. **LLM 없음, 순수 함수** | 결정론 |
| 실행 | 워커 6개 | 자기 출력 키만 쓴다. 워커끼리 직접 통신 없음 — 전부 Supervisor 층으로 복귀 | — |

- **판정과 게이트를 물리적으로 다른 노드로 둔다** — 교안 부록 B "판정은 확률에, 게이트는 결정론으로" ·
  "`judge_node` → `gate()`" · "결정론 층이 먼저 거르고 통과된 것만 Judge 로". 수업 노트북 `07-Supervisor-Advanced` 의
  `RelevanceChecker → Supervisor` 구조와 같다.
- **매 턴 워커 1명** — 교안 p.71 · p.123: Supervisor 는 "State 기준으로 1개씩 순차 호출". RAG 때의 3개 병렬 fan-out 은 없앤다.
- **종료는 코드가 강제** — `max_steps` · 셀별 재작업 ≤2 · 평가 루프 ≤2 · `LLM_BUDGET` · `recursion_limit` (교안 p.167).

### 실무 관점 — 이 설계는 Workflow 인가 Agent 인가

교안 p.57: **Workflow**(코드가 경로 통제) ↔ **AI Agentic Workflow**(고정된 바깥 구조 + 그 안에서 모델의 자율 선택) ↔ **Agent**(모델이 런타임 흐름 결정).
"Production 에서는 Agent 보다 Workflow 비중이 높다" · "Agent ≠ Loop — 루프가 있어도 구조가 코드로 고정이면 Workflow" ·
"Enterprise 는 Hybrid, Guardrailed Agent" (p.10–11).

우리 설계는 **AI Agentic Workflow (Guardrailed)** 에 둔다.

| 코드가 고정하는 것 (Workflow 골격 · Harness) | 모델이 런타임에 정하는 것 (Agentic) |
|---|---|
| 그래프 토폴로지(허브), 4관점 × 2기술 = 8셀, 보고서 뒤 평가 노드 | 셀별 **근거 충분 여부** → 어느 워커가 몇 번 재작업될지 |
| 규칙 검사 기준(건수 · 반대 근거 · 출처 다양성 · 실패 기록) | 재작업 때 던질 **보강 질의**(`hint_query`) |
| 게이트 규칙과 모든 상한 | 보고서 미달 사유 → 보고서 재작성인지, 특정 관점 재조사인지 |

⚠️ 규칙만으로 라우팅하면 "Agent ≠ Loop" 기준에 따라 **그냥 Workflow** 로 읽힌다. LLM 충분성 Judge 가 있어야
"Supervisor 가 충분한가를 스스로 판단"(p.70 Supervisor vs Router)이 성립한다. 그래서 충분성은 **Hybrid** 다.

---

## 1. 누가 무엇을

| | 이름 | 한 줄 | 소유 파일 |
|---|---|---|---|
| **A** | 박유진 | 충분성 판정 · 관측성 · 기술 조사 | `graph/sufficiency.py` · `graph/observe.py` · `prompts/sufficiency_judge.md` · `rag/*` · `agents/tech_research.py` · `prompts/tech_research.md` `rag_*.md` · `experiments/` |
| **B** | 심준용 | 시장 · 이해관계자 · 보고서 · 실증 자료 | `agents/market.py` `stakeholder.py` `report.py` `report_render.py` · `prompts/market.md` `stakeholder.md` `report.md` · `prompts/rubrics/4.2` `4.3` · `outputs/report/` · `docs/tracing/` |
| **C** | 민영은 | 품질 평가 노드 · 도메인 · 종합 | `agents/evaluator.py` · `prompts/evaluator.md` · `agents/domain.py` `synthesis.py` · `prompts/domain.md` `synthesis.md` `neutrality_judge.md` · `prompts/rubrics/4.1` `4.4` `4.5` |
| **D** | 황재원 | Supervisor · State · Graph · 통합 | `graph/supervisor.py` `state.py` `build.py` `safe.py` · `app.py` · README 취합 |

공용 (바꾸기 전에 슬랙): `agents/_common.py` · `config/` · `scripts/` · `pyproject.toml` · CI · `tests/fixtures/`

> RAG 때와 바뀐 점: `agents/report.py` 가 D → **B** 로 이동 (D 가 Supervisor · State · 체크포인터를 다 맡아서).
> `graph/dispatcher.py` 는 `graph/supervisor.py` 로 대체되고 삭제된다 (D).

---

## 2. 파트별 할 일

### A 박유진 — 충분성 판정 (`assess` 노드) · 관측성 · 기술 조사

| 파일 | 함수 | 하는 일 |
|---|---|---|
| `graph/sufficiency.py` | `check_rules(state) -> dict[cell, CellVerdict]` | **결정론 층.** 8셀(관점 × 기술)마다 아래 「충분성 기준」. 실패하면 Judge 를 부르지 않는다 |
| 〃 | `assess(state) -> dict` | 노드. 규칙 통과 셀만 `llm("judge")` + `prompts/sufficiency_judge.md` 로 4기준 판정 (Pydantic structured output). 결과를 `sufficiency` 에 기록만 — **`next` 는 쓰지 않는다**. 재판정은 방금 실행된 워커의 셀만(재작업이면 그 기술만), Judge 호출 실패 시 규칙 판정만 남긴다 |
| `graph/observe.py` | `new_trace_id()` · `log_decision(state, node, decision, reason)` | 결정 로그를 `outputs/decisions.jsonl` + LangSmith run metadata 로 외부 적재. `{trace_id, node, decision, reason, ts}` |
| `agents/tech_research.py` | `run(state)` | `rework_request` 가 자기 것이면 **해당 기술만** `hint_query` 로 보강 검색 |
| `rag/rag_node.py` | `ask()` | `retrieval_log` 를 State 대신 `outputs/retrieval_log.jsonl` (trace_id 포함) 로 — 지속성 비용 |

**충분성 기준** — 셀 하나 = `"market:InfiniGen"` 처럼 `"{worker}:{tech}"`. 위에서부터 처음 걸리는 항목의 `gap` 으로 부족 판정.
원칙: **공통 기준 하나 + 자료 구조가 달라서 공통 기준이 성립하지 않는 곳만 예외.** 숫자는 데이터에 맞추지 않고 설명 가능한 원칙으로 정한 뒤, 데이터로는 이상 동작만 확인한다.

| 기준 | 적용 | 값 | `gap` | 근거 |
|---|---|---|---|---|
| 워커 실패 | 전 관점 | `node_status == "failed"` 또는 safe fallback | `failed` | 실패 기록은 근거가 아니다 |
| 등급 과반이 "근거 없음" | 시장 · 이해관계자 · 도메인 | 하위 항목 과반 | `no_evidence` | — |
| 근거 건수 | 전 관점 | `[논문]` · `[웹]` **≥3** (`[추론]` · Faithfulness 미통과 메모 제외) | `evidence` | 출처 없는 문장은 근거가 아니다 (#64 리뷰) |
| 반대 근거 | 시장 · 이해관계자 · 도메인 | **≥2** — `[추론]` 으로 끝나는 표시("반대 근거 확보 실패 [추론]")는 세지 않음 | `negatives` | RAG 확증편향 방지 장치 7 그대로. 질은 Judge 기준 3 |
| 출처 다양성 | 시장 · 이해관계자 (웹만 쓰는 관점) | 출처 **≥2곳**, 한 출처가 **과반(>50%)이면 부족** | `source_bias` | 단일 출처 편중 = 확증편향 (노션 「편향 통제」) |
| 웹 근거 | 도메인 | `[웹]` 근거 **≥1** — 반례인지(지지 근거가 아닌지)는 규칙이 못 보고 **Judge 기준 3** 이 본다 | `counter_example` | 도메인 = 논문 사실 추출 + 웹 반례 (RAG 설계서 4.4 · `domain.yaml`) |
| 수치 · 한계 | 기술 조사 | 수치 ≥1 · 한계 ≥1 (한계가 반대 근거 역할) | `numbers` · `limitations` | 검색 대상이 그 기술 논문 1편 → **출처 다양성은 보지 않는다** |
| LLM Judge (규칙 통과 셀만) | 전 관점 | ① 주장↔근거 대응 ② 관점 적합성 ③ 반대 근거의 실질 ④ 우열 판정 없음 | `unsupported` · `off_topic` · `negatives` · `bias` | 형식은 규칙이, 내용은 Judge 가 |

지난 RAG 제출 실행 데이터로 확인: 8셀 중 부족 5셀 (시장 InfiniGen github 88% · 이해관계자 KIVI 자기 논문 75% · 이해관계자 InfiniGen 근거 없음 · 도메인 KIVI 웹 근거 0 · 시장 KIVI Judge off_topic).
일부러 기준을 낮추거나 올리지 않는다 — 첫 실제 통합 실행 뒤 **오판정이 있을 때만** 한 번 조정하고, 바꾸면 README Lessons Learned 에 한 줄.

### B 심준용 — 시장 · 이해관계자 재작업, 보고서, 실증 자료

- `market.py` · `stakeholder.py`: `rework_request = {worker, tech, gap, hint_query}` 가 자기 것이면 **그 기술만** 다시 돌려 반환 (`*_eval` 은 기술 단위 병합 reducer 라 다른 기술 결과는 남는다). 기존 "반대 근거만 검색" 모드는 `gap == "negatives"` 로 흡수
- `report.py`: `eval_result.feedback` 이 있으면 반영해 재작성. 본문은 `outputs/report/report.md` 로 쓰고 State 에는 **`report_uri` 만** 반환. **A4 10장 이하** — 넘으면 4장 관점별 서술을 줄인다. SUMMARY · REFERENCE 필수
- 실증: LangSmith 에서 **재작업 · 평가 루프가 한 번 이상 찍힌 실행**을 골라 `docs/tracing/tracing-1.png` 부터 캡처. README Run Record 표 갱신

### C 민영은 — 품질 평가 노드 (`evaluator`) · 도메인 · 종합

- `agents/evaluator.py` — 보고서 **뒤**. 노션 4항목, 방식은 3안 Hybrid:

  | 항목 | 규칙 (먼저) | LLM Judge (규칙 통과 시) |
  |---|---|---|
  | Groundedness | 판단 문장 출처 태그 비율 · `[추론]` ≤10% · REFERENCE ↔ 본문 인용 대응 | 주장 샘플 ↔ evidence 대조 |
  | 중립성 | 금지어("더 낫다" · "권장" · "선택해야" · "우수") | 우열 판정 문맥인지 |
  | 편향 통제 | 단일 출처 비중 ≤40% · 셀별 반대 근거 존재 | — |
  | 관점 커버리지 | 4관점 × 2기술 섹션 · SUMMARY · REFERENCE 존재 | — |

  결과는 `eval_result` 에 **기록만** — 다음 경로는 D 의 게이트가 정한다. 실패 항목마다 `target`(`"report"` 또는 `"{worker}:{tech}"`)과 `feedback` 을 남긴다
- `synthesis.py` 안 중립성 Judge 를 evaluator 로 합칠지 **오전에 결정** (합치면 중복 제거, 두면 수정량 최소)
- `domain.py`: `rework_request` 처리 (A · B 와 같은 방식)

### D 황재원 — Supervisor · State · Graph · 통합

- `graph/supervisor.py`: `supervisor(state)` 노드 = **순수 함수**. `sufficiency` · `eval_result` · 시도 횟수 · 상한만 보고 `next` 1개 + `rework_request` + `step_count += 1`. 사유는 `observe.log_decision()`. `route(state)` 는 `state["next"]` 반환만
  - 우선순위 (State 에서 계산): 상한 초과 → `end_with_warning` · 미수집 셀(선행 조건 충족한 것) → 해당 워커 · 부족 셀(재작업 <2) → 재작업 · 전부 충분/소진 → synthesis → report · 평가 fail(<2) → target · 평가 pass → END
- `graph/state.py`: 아래 4절 계약대로 재편
- `graph/build.py`: 토폴로지 (0절 그림) + 체크포인터(SqliteSaver, `thread_id = trace_id`)
- `graph/safe.py`: 실패 시 `node_status[name] = "failed"` · `last_error` · `errors` 도 함께 반환 — **fallback 이 정상 결과처럼 보이지 않게** (교안 함정: except 가 오류를 삼키면 fallback 이 정상처럼 보인다)
- `end_with_warning` 노드: 상한 소진 시 `status = "FAILED"` 대신 미달 항목을 보고서 한계점에 남기고 종료 — 트레이스에 이름으로 찍힌다
- `app.py`: `--resume <trace_id>` 로 체크포인트 재개
- 오전에 **강사 확인 2건**: ① Supervisor 에 LLM 이 꼭 필요한가 ② 우선순위 규칙이 "순서 하드코딩"으로 읽히는가

---

## 3. 의존 그래프

```
D  state.py 계약 PR ──┬───────────────┬───────────────┬──────────────┐
                      ▼               ▼               ▼              ▼
A  sufficiency ──► D supervisor ◄── C evaluator    B 워커 재작업   A·C 워커 재작업
   (assess)            │  (gate)        ▲
                       ▼                │
                    D build.py ──► B report ──┘
                       │
                       ▼
                 통합 실행 → B 트레이스 캡처 → D README 취합
```

| 기다리는 쪽 | 기다리는 것 | 막힐 때 우회 |
|---|---|---|
| 전원 | D `state.py` 계약 | 4절 표의 키 이름으로 먼저 개발 |
| D supervisor | A `sufficiency` · C `eval_result` 형식 | 형식만 맞춘 stub 을 fixtures 에 |
| B report | C `eval_result.feedback` | feedback 없는 경로부터 |
| B 캡처 | 통합 실행 성공 | — 마감 2시간 전까지 반드시 |

**병목은 D `state.py` 와 A `assess` · C `evaluator` 의 출력 형식.** 셋이 오전 첫 PR.

---

## 4. 인터페이스 계약 — `graph/state.py` 에 이대로 들어간다

```python
# ── 페이로드 (작업 결과) ─────────────────────────────
domain, selected                                  # 입력
tech_summary:     Annotated[dict[Tech, TechSummary], merge_by_tech]
market_eval:      Annotated[dict[Tech, Eval],        merge_by_tech]
stakeholder_eval: Annotated[dict[Tech, Eval],        merge_by_tech]
domain_eval:      Annotated[dict[Tech, DomainEval],  merge_by_tech]
trl_estimate, synthesis
report_uri: str                                   # 본문은 파일 — State 는 참조만
citations:  Annotated[list[Ref], operator.add]

# ── 판정 (Judge 노드가 쓰고, Gate 가 읽는다) ────────
sufficiency: Annotated[dict[str, CellVerdict], merge]   # "market:InfiniGen" → 아래
eval_result: EvalResult | None

# ── 제어 (Gate 만 쓴다 · safe 는 node_status/errors 만) ─
trace_id: str                                     # = 체크포인터 thread_id = LangSmith metadata
next: str                                         # 매 턴 1개
rework_request: ReworkRequest | None
step_count: Annotated[int, operator.add];  max_steps: int
retry: Annotated[dict[str, int], merge]           # "market:InfiniGen" → 횟수
eval_attempts: int
llm_calls: Annotated[int, operator.add]
status: Literal["RUNNING", "SUCCESS", "FAILED", "INTERRUPTED"]
node_status: Annotated[dict[str, str], merge]     # worker → "ok" | "failed"
errors: Annotated[list[ErrorRecord], operator.add]
last_error: ErrorRecord | None                    # {node, type, message, ts}
```

| 이름 | 형식 |
|---|---|
| `CellVerdict` | `{rule: "pass"\|"fail", judge: "sufficient"\|"insufficient"\|None, gap: str, hint_query: str, reason: str}` |
| `ReworkRequest` | `{worker, tech, gap, hint_query}` — 워커는 `worker` 가 자기 이름일 때만 읽는다 |
| `EvalResult` | `{passed: bool, items: {groundedness, neutrality, bias, coverage: {passed, score, reason}}, targets: list[str], feedback: str}` |
| 워커 반환 | 자기 출력 키 + `citations` + `llm_calls` 만. **`next` · `retry` · `sufficiency` · `eval_result` 는 절대 쓰지 않는다** |

README State Schema 7항목과의 대응: 제어 vs 페이로드 = 위 3구역 · 관측성 = `observe.py` 외부 적재 · 지속성 = `report_uri` · `retrieval_log.jsonl` ·
상관 = `trace_id` · 재개 = 체크포인터 + `node_status` · `last_error` · 동시 처리 = `merge_by_tech` 등 reducer · 종료 = `max_steps` · `retry` · `eval_attempts` · `LLM_BUDGET`

---

## 5. 초기 이슈

| 파트 | 제목 | type |
|---|---|---|
| D | `[FEAT] State 계약 재편 — 제어/판정/페이로드 · trace_id · reducer` | feat |
| D | `[FEAT] Supervisor 게이트 + route() + end_with_warning — dispatcher 대체` | feat |
| D | `[FEAT] 체크포인터 · safe node_status · app.py --resume` | feat |
| A | `[FEAT] assess 노드 — 충분성 규칙 → LLM Judge` | feat |
| A | `[FEAT] observe — trace_id · 결정 로그 · retrieval_log 외부화` | feat |
| A | `[FEAT] tech_research rework_request 처리` | feat |
| B | `[FEAT] market · stakeholder rework_request 처리` | feat |
| B | `[FEAT] report — feedback 반영 · report_uri · 10장` | feat |
| C | `[FEAT] evaluator 노드 — 4항목 규칙 → LLM Judge` | feat |
| C | `[FEAT] domain rework_request · 중립성 Judge 통합 여부` | feat |
| B | `[DOCS] LangSmith tracing 캡처 · Run Record` | docs |
| D | `[DOCS] README — Pattern · 동적 처리 · State Schema 7항목` | docs |

---

## 5-1. 트레이스 캡처 규칙 — 전원 같은 방식으로

노션 제출물 2번 「LangSmith Tracing — 동적 처리 확인 목적」. 채점(동적 동작 실증 20 · 재현성 10)은 **캡처 · 코드 · README 가 같은 실행을 가리키는지** 본다.

| 무엇 | 규칙 |
|---|---|
| 이름 | LangSmith 프로젝트 · 태그 · run_name 전부 **`kv-cache-agent`**. 각자 `.env` 의 `LANGSMITH_PROJECT=kv-cache-agent` (`.env.example` 참고). RAG 과제 `kv-cache-eval` 과 섞지 않는다 |
| 실행 하나 | 캡처는 **한 실행(trace_id) 것만**. `app.py` 가 찍는 `trace_id=…` 를 LangSmith 검색창에 태그로 넣으면 그 실행만 나온다. 여러 실행을 섞어 붙이지 않는다 |
| 고르는 기준 | `uv run python -m graph.run_summary <trace_id>` 로 확인 — **셀 재작업 ≥1 · 평가 루프 ≥1** 이 찍힌 실행. 일부러 기준을 낮춰 만들지 않는다 (CONTRIBUTING 6절) |
| 파일 | `docs/tracing/tracing-1.png`, `tracing-2.png` … — **시간 순서**. 경로가 길면 여러 장으로 나눈다 (노션) |
| 보여야 할 것 | ① 전체 트리 (supervisor → 워커 → assess → supervisor 반복) ② **재작업** 한 장면 — supervisor run 의 metadata `decision` · `reason` ("부족 셀 … 재작업 1/2") ③ **평가 루프** — evaluator → supervisor → report 재작성 ④ 종료 (END 또는 end_with_warning) |
| README 연결 | Run Record 표에 같은 `trace_id` · `run_summary --md` 결과 · 캡처 파일 목록을 적는다 |
| 담당 | 캡처 B 심준용 · 실행 고르기 A 박유진 (`run_summary`) · README D 황재원 |

---

## 6. 시간표 (DAY 2)

| | A 박유진 | B 심준용 | C 민영은 | D 황재원 |
|---|---|---|---|---|
| 오전 1 | **전원 30분: 4절 계약 확정** | | | |
| 오전 2 | `check_rules` + `assess` stub PR | 워커 2개 rework 처리 | `evaluator` 규칙 층 + stub PR | **`state.py` PR** → supervisor stub · 강사 확인 |
| 점심 직후 | **전원: 첫 통합 실행 — stub 이어도 END 까지 가는지** | | | |
| 오후 1 | LLM Judge · `observe` | `report` feedback · 10장 | LLM Judge 층 · domain rework | 체크포인터 · `end_with_warning` |
| 오후 2 | Retrieval 지표 README | **재작업 · 평가 루프 찍힌 실행 캡처** | 품질 평가 README | README 취합 |
| 마감 1h 전 | **전원: PDF 10장 확인 · zip · Slack 제출** | | | |

---

## 7. README Contributors (PM · PL 표기 금지 — 노션)

- 박유진 : 근거 충분성 판정(규칙 + LLM Judge), 관측성(trace_id · 결정 로그), 기술 조사 Agent, RAG Pipeline
- 심준용 : 시장 · 이해관계자 Agent 재작업, 보고서 생성 Agent, LangSmith Tracing 실증
- 민영은 : 품질 평가 노드(Groundedness · 중립성 · 편향 · 커버리지), 도메인 · 종합 Agent
- 황재원 : Supervisor 게이트 · State Schema · Graph · 체크포인터, README 취합
