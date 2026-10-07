# 협업 규칙 — Agent 과제 (`agent-supervisor` 브랜치)

> 이슈 → 브랜치 → PR → 리뷰 → 머지. 예외 없이 이 순서로 갑니다.
> **Agent 과제의 모든 PR 은 무조건 `agent-supervisor` 로 머지합니다.** 예외 없음.
> `main` 은 RAG 과제 제출본으로 **동결** — 어떤 PR 도 `main` 으로 보내지 않습니다.
> base 가 `main` 인 PR 은 CI `base-branch` 검사가 **빨간불**로 막습니다 (유일한 예외: 템플릿 반영 #50).
> 분담 · 계약 · 시간표는 [docs/ROLES.md](docs/ROLES.md).

---

## 0. 이 저장소의 특수 사정

개발 시간이 **하루**(DAY 2 퇴근 전 마감)이고, 워커 6개 · 판정 노드 2개 · Supervisor 가 **State 하나**(`graph/state.py`)로 묶여 있습니다.
그래서 아래 세 가지가 다른 어떤 규칙보다 중요합니다.

1. **State 키 이름·타입은 `graph/state.py` 가 유일한 정본** — 워커가 임의로 키를 추가하지 않는다. 필요하면 D 에게 이슈
2. **자기 디렉토리만 고친다** — 각 파일 첫 줄 docstring 의 `[소유: X]` 확인. 남의 파트는 이슈로 넘긴다
3. **판정과 게이트를 섞지 않는다** — `assess` · `evaluator`(Judge) 는 판정 결과만 State 에 쓰고, `next` 는 `supervisor`(Gate, 순수 함수) 만 쓴다.
   워커는 `next` · `retry` · `sufficiency` · `eval_result` 를 절대 반환하지 않는다 (교안 부록 B — 판정은 확률, 게이트는 결정론)

워커 인터페이스는 전부 `run(state) -> dict` 이고 **자기 출력 키 + `citations` + `llm_calls`** 만 반환합니다.
`rework_request` 가 있으면 `worker` 가 자기 이름일 때만 읽고, **그 기술만** 다시 돌립니다 (`*_eval` 은 기술 단위 병합 reducer).
충돌이 난다면 거의 `state.py` · `pyproject.toml` · `README.md` 입니다.

### 1회 설정 — 각자 자기 PC 에서 한 번만

```bash
uv sync --group dev              # .venv 생성 (Python 3.11)
cp .env.example .env             # OPENAI_API_KEY, TAVILY_API_KEY, LANGSMITH_API_KEY 입력
bash scripts/fetch_papers.sh     # data/papers/kivi.pdf, infinigen.pdf (커밋 안 함)
uv run python -m rag.indexing    # data/index/ (A 가 머지한 뒤부터 가능)
uv run pytest                    # LLM 없이 도는 테스트
```

- 패키지 추가는 `pip` 말고 `uv add <pkg>` → `pyproject.toml` · `uv.lock` 함께 커밋
- 임베딩(BGE-M3, 2.27GB)은 첫 실행 때 HF 캐시로 내려받음. 이미 있으면 0분

---

## 1. 파트 소유

| | 이름 | 담당 | 소유 디렉토리 |
| --- | --- | --- | --- |
| **A** | 박유진 | 충분성 판정(`assess`) · 관측성 · 기술 조사 · RAG | `graph/sufficiency.py` · `graph/observe.py` · `prompts/sufficiency_judge.md` · `rag/` · `agents/tech_research.py` · `experiments/` · `prompts/rag_*.md` · `prompts/tech_research.md` |
| **B** | 심준용 | 시장 · 이해관계자 · 보고서 · 트레이스 캡처 | `agents/market.py` · `agents/stakeholder.py` · `agents/report.py` · `agents/report_render.py` · `prompts/market.md` · `prompts/stakeholder.md` · `prompts/report.md` · `prompts/rubrics/4.2` `4.3` · `outputs/report/` · `docs/tracing/` |
| **C** | 민영은 | 품질 평가(`evaluator`) · 도메인 · 종합 | `agents/evaluator.py` · `prompts/evaluator.md` · `agents/domain.py` · `agents/synthesis.py` · `prompts/domain.md` · `prompts/synthesis.md` · `prompts/neutrality_judge.md` · `prompts/rubrics/4.1` `4.4` `4.5` |
| **D** | 황재원 | Supervisor(게이트) · State · Graph · 통합 | `graph/supervisor.py` · `graph/state.py` · `graph/build.py` · `graph/safe.py` · `app.py` · README 취합 |

> RAG 과제 대비 변경: `agents/report.py` · `prompts/report.md` 가 D → B, `graph/` 안에서도 파일 단위로 소유가 갈린다
> (`sufficiency.py` · `observe.py` 는 A). `graph/dispatcher.py` 는 `supervisor.py` 로 대체 후 삭제 (D).

`config/` · `scripts/` · `pyproject.toml` · `.env.example` · CI · `README.md` 는 공용입니다. 바꾸기 전에 슬랙에 공유하세요.
누가 무엇을 기다리는지는 [docs/ROLES.md](docs/ROLES.md) 의존 그래프 참고.

**남의 파트에 변경이 필요하면 직접 고치지 말고 이슈를 만들어 소유자에게 넘깁니다.**

---

## 2. 브랜치 전략

노션 명세: "**Branch 로 기존 작업과 구분**", GitHub 링크는 그대로. 그래서 `main`(RAG 제출본)은 건드리지 않고
`agent-supervisor` 를 Agent 과제의 기준 브랜치로 씁니다. 그 아래는 RAG 때와 같은 **기준 브랜치 + 작업 브랜치** 2단계.

```
main ────●  (RAG 제출본 · 동결)
          \
agent-supervisor ──●────────●────────●──▶   (항상 python app.py 가 도는 상태 · 제출 링크)
                    \      /  \     /
                     feat/50  feat/52        (이슈 1개 = 브랜치 1개, base = agent-supervisor)
```

- 제출할 GitHub 링크: `https://github.com/TurtleNeck-Crew-RAG/kvcache-agentic-rag/tree/agent-supervisor`
- 작업 브랜치 이름 규칙은 아래 그대로. **저장소 기본 브랜치가 `agent-supervisor`** 라서 PR base · 새 브랜치 출발점이 기본으로 여기 잡힙니다 (2026-10-07 변경, 마감 후 `main` 으로 되돌림)

### 네이밍 규칙

```
<type>/<이슈번호>-<slug>
```

| type | 언제 | 예시 |
| --- | --- | --- |
| `feat` | 새 기능 (워커, RAG 단계, 그래프) | `feat/3-stakeholder-worker` |
| `fix` | 버그 수정 | `fix/11-retry-counter-overwrite` |
| `chore` | 설정, CI, 의존성 | `chore/5-add-flagembedding` |
| `docs` | 문서·설계서·README 만 수정 | `docs/8-readme-tech-stack` |
| `refactor` | 동작 변화 없는 정리 | `refactor/14-prompt-loader` |
| `exp` | 실험·실측 (`experiments/`) | `exp/6-sparse-bm25-vs-m3` |

- slug 는 **영문 소문자 + 하이픈**. 한글·공백·대문자 금지.
- 이슈 번호는 필수. 번호 없는 브랜치는 리뷰하지 않습니다.
- 한 브랜치에 한 가지 일만. 커지면 이슈를 쪼개세요.

> 이슈 페이지 오른쪽 **Development → Create a branch** 를 쓰면 브랜치가 이슈에 자동 연결됩니다.
> 기본 브랜치가 `agent-supervisor` 라 따로 고를 필요 없이 여기서 갈라집니다.
> GitHub 이 제안하는 이름(`3-feat-...`)은 그 팝업에서 `feat/3-stakeholder-worker` 로 **직접 고쳐서** 만드세요.
> PR 머지 시 `closes #N` 으로 이슈가 자동으로 닫힙니다.

---

## 3. 작업 순서

```bash
# 1) 이슈 생성 (GitHub 웹에서 템플릿 선택) → 이슈 번호 확인 (#3)

# 2) 최신 agent-supervisor 받기
git fetch origin
git switch agent-supervisor
git pull origin agent-supervisor

# 3) 브랜치 생성
git switch -c feat/3-stakeholder-worker

# 4) 작업 → 테스트 → 커밋
uv run pytest
git add .
git commit -m "feat(agents): 이해관계자 워커 — 찬반 각 2건 강제"

# 5) 푸시
git push -u origin feat/3-stakeholder-worker

# 6) GitHub에서 PR 생성 — base: agent-supervisor ← compare: feat/... (템플릿 자동 삽입됨) → 리뷰 요청

# 7) 승인 후 Squash and merge → 브랜치 삭제
```

---

## 4. 커밋 컨벤션

```
<type>(<scope>): <한글 요약>
```

scope 는 디렉토리명: `rag` `agents` `graph` `prompts` `config` `docs` `exp` `ci`

| type | 의미 |
| --- | --- |
| `feat` | 기능 추가 |
| `fix` | 버그 수정 |
| `chore` | 설정, 빌드, 패키지 |
| `docs` | 문서 |
| `refactor` | 리팩터링 |
| `test` | 테스트 |
| `exp` | 실험 결과·스크립트 |

```bash
git commit -m "feat(rag): EnsembleRetriever sparse+dense 0.5/0.5 k=4"
git commit -m "fix(graph): retry reducer 가 키를 덮어쓰던 문제"
git commit -m "exp(rag): BGE-M3 sparse vs BM25 20문항 Hit@4 비교"
git commit -m "docs: README Tech Stack 에 Hit Rate/MRR 기재"
```

**scope 를 붙이는 이유** — 워커별로 파일이 갈려 있어서 scope 만 보면 누구 작업인지 압니다.

### gitmoji는 **선택**입니다

쓰고 싶으면 type 앞에 붙이세요. 안 써도 리뷰에서 지적하지 않습니다.

| gitmoji | type | | gitmoji | type |
| --- | --- | --- | --- | --- |
| ✨ | feat | | ♻️ | refactor |
| 🐛 | fix | | ✅ | test |
| 📝 | docs | | 🔧 | chore |
| 🧪 | exp | | | |

> 딱 하나만 맞춰주세요: **`<type>:` 은 반드시 있어야 합니다.**

---

## 5. PR 규칙

- 제목: `[FEAT] 이해관계자 워커 — 찬반 각 2건 강제` (이슈 제목과 맞추면 편합니다)
- 본문의 `closes #3` 을 **반드시** 채우기 → 머지 시 이슈 자동 종료 (기본 브랜치 `agent-supervisor` 로 가는 PR 이라 동작)
- **base 는 무조건 `agent-supervisor`.** `main` 으로 열었으면 PR 화면 제목 옆 `Edit` → base 드롭다운에서 바꾸면 됩니다 (새로 열 필요 없음).
  CI `base-branch` 검사가 실패하면 base 가 `main` 이라는 뜻입니다
- 이슈 · PR 템플릿은 GitHub 이 기본 브랜치에서 읽습니다 — 지금은 `agent-supervisor` 의 `.github/` 가 쓰입니다.
  `main` 의 `.github/` 는 #51 에서 넣은 같은 안내의 사본 (마감 후 기본 브랜치를 `main` 으로 되돌려도 안내가 남게)
- 리뷰어 **최소 1명** 승인 후 머지
- 머지 방식: **Squash and merge**
- 머지 후 원격 브랜치 삭제
- 머지는 **리뷰어 승인 후** 합니다. 작성자가 못 하는 상황이면 승인한 팀원 누구든 머지해도 됩니다. 승인 없이 머지하지 않기 — 리뷰가 급하면 슬랙에서 부르기

### PR을 잘게 쪼개주세요

**"워커 하나 = PR 하나", "RAG 단계 하나 = PR 하나"** 수준으로 올려야
리뷰가 밀리지 않고 마지막에 충돌이 안 납니다.

### CI 를 통과해야 머지할 수 있습니다

PR 을 올리면 `검사` 가 자동으로 돕니다.

| 검사 | 실패하면 |
| --- | --- |
| API 키 하드코딩 (`sk-`, `tvly-`, `lsv2_`) | ❌ 실패 — `.env` 로 옮기기 |
| 커밋 금지 파일 (`.env`, `data/papers/*.pdf`, `data/index/`, `reference/`) | ❌ 실패 |
| `ruff check` (문법 · import 순서) | ❌ 실패 — `uv run ruff check --fix .` |
| `pytest tests/` (LLM 없이 도는 것만) | ❌ 실패 |

---

## 6. 🚨 하지 말아야 할 것

| 대상 | 이유 |
| --- | --- |
| **`graph/state.py` 에 키 추가·이름 변경** | 전 워커가 깨집니다. D 에게 이슈 |
| **남의 디렉토리 수정** | 소유자에게 이슈로 넘기세요 |
| **두 기술을 같은 프롬프트에 넣기** | 편향 방지 장치 2 위반 (설계서 5.5). 기술별로 따로 호출 |
| **풀 밖 문서를 `data/papers/` 에 넣고 인덱싱** | 과제 명세 위반. 반례·시장 자료는 웹검색으로 |
| **`[추론]` 태그 없이 판단 문장 쓰기** | 태그 비율이 보고서 한계점 수치입니다 |
| **API 키 코드 직접 입력** | `.env` + `load_dotenv`. CI 가 막습니다 |
| **`next` · `retry` · `sufficiency` · `eval_result` 를 워커에서 반환** | 게이트만 쓰는 키. 워커가 쓰면 라우팅이 재현되지 않는다 |
| **Judge 노드(`assess` · `evaluator`) 안에서 다음 노드 결정** | 판정과 게이트가 섞이면 결정론 경계가 사라진다 (교안 부록 B) |
| **워커 `except` 에서 오류를 삼키고 정상 결과처럼 반환** | `safe.py` 가 `node_status = "failed"` 로 남기게 둔다. 삼키면 충분성 판정이 실패를 "충분"으로 본다 |
| **트레이스용으로 충분성 기준을 일부러 낮추거나 실패를 연출** | 재현성 항목에서 제출 trace 와 비교된다 |
| **`main` 으로 PR · push** | `main` 은 RAG 제출본. 아래 ⚠️ 참조 |

> ⚠️ **`main` 보호**: CI `base-branch` 검사가 base 가 `main` 인 PR 을 빨간불로 표시합니다.
> 다만 빨간불이어도 머지 버튼 자체가 잠기지는 않으니, **빨간불 PR 은 머지하지 않습니다.**

---

## 7. 설계와 코드가 어긋나면

Agent 과제의 설계 정본은 [docs/ROLES.md](docs/ROLES.md) 0절(패턴) · 4절(State 계약)과 README 의 State Schema 7항목입니다.
`docs/설계서.md` 는 RAG 과제 정본으로 그대로 둡니다. 개발하다 바꿔야 할 게 생기면:

1. `[CHORE] 설계 변경` 이슈로 올리고 조원 합의
2. ROLES.md 4절과 `graph/state.py` 를 **같은 PR** 에서 고친다
3. 바뀐 이유를 README **State Schema** 또는 **Lessons Learned** 에 한 줄

채점은 "README 에 쓴 설계 근거가 코드에 구현됐는가"(State Schema 20점)를 봅니다. README 와 코드가 다르면 감점입니다.

---

## 8. 충돌이 났다면

```bash
git fetch origin && git rebase origin/agent-supervisor    # 내 브랜치를 최신 agent-supervisor 위로
```

`state.py` 충돌이면 D 의 버전을 살리고 내 워커를 그 키 이름에 맞춥니다.
`pyproject.toml` · `uv.lock` 충돌이면 양쪽 의존성을 다 살린 뒤 `uv lock` 을 다시 돌립니다.

---

## 9. 부록 — 이슈 템플릿 YAML 읽는 법

`.github/ISSUE_TEMPLATE/*.yml` 은 GitHub 의 **Issue Forms** 형식입니다.

```yaml
name: "✨ Feature"          # 이슈 만들기 화면의 카드 제목
description: "새 기능 추가"   # 카드 부제
title: "[FEAT] "           # 이슈 제목 기본값
labels: ["feature"]        # 자동으로 붙는 라벨

body:                      # 위에서부터 순서대로 렌더링
  - type: input            # 한 줄 입력
  - type: textarea         # 여러 줄 입력
  - type: dropdown         # 선택지 (options 필요)
  - type: checkboxes       # 체크박스 묶음
  - type: markdown         # 안내문 (제출 내용에 안 들어감)
```

| 자주 틀리는 것 | 결과 |
| --- | --- |
| 들여쓰기에 **탭** 사용 | 파싱 실패 → 템플릿이 아예 안 보임 |
| `label` 없이 `description` 만 씀 | `label` 은 필수 |
| `placeholder` 를 기본값으로 착각 | 실제로 채우려면 `value` |
| `config.yml` 을 템플릿으로 착각 | 템플릿이 아니라 **선택 화면 설정** |

> **템플릿은 `main` 에 있어야 보입니다.** PR 브랜치에만 있으면 이슈 생성 화면에 안 뜹니다.
> **PR 템플릿은 저장소 화면 어디에도 따로 안 보입니다.** 브랜치를 푸시하고 `Compare & pull request` 를 눌렀을 때 PR 본문에 자동으로 채워지는 게 전부입니다.
> 템플릿 PR 을 가장 먼저 머지하고 `…/issues/new/choose` 에서 확인하세요.
> `config.yml` 의 링크는 `TurtleNeck-Crew-RAG/kvcache-agentic-rag` 기준입니다.
