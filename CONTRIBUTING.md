# 협업 규칙

> 이슈 → 브랜치 → PR → 리뷰 → 머지. 예외 없이 이 순서로 갑니다.
> `main` 에 직접 push 금지.

---

## 0. 이 저장소의 특수 사정

개발 시간이 **하루**이고, 워커 6개가 **State 키 하나**(`graph/state.py`)로 묶여 있습니다.
그래서 아래 두 가지가 다른 어떤 규칙보다 중요합니다.

1. **State 키 이름·타입은 `graph/state.py` 가 유일한 정본** — 워커가 임의로 키를 추가하지 않는다. 필요하면 D 에게 이슈
2. **자기 디렉토리만 고친다** — 각 파일 첫 줄 docstring 의 `[소유: X]` 확인. 남의 파트는 이슈로 넘긴다

워커 인터페이스는 전부 `run(state) -> dict` 이고 **자기 출력 키만** 반환합니다 (설계서 5.2).
그래서 워커끼리는 파일이 겹칠 일이 없고, 충돌이 난다면 거의 `state.py` · `pyproject.toml` · `README.md` 입니다.

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
| **A** | 박유진 | RAG 파이프라인 · 기술 조사 · 임베딩 실측 | `rag/` · `agents/tech_research.py` · `experiments/` · `prompts/rag_*.md` · `prompts/tech_research.md` |
| **B** | 심준용 | 시장 · 이해관계자 | `agents/market.py` · `agents/stakeholder.py` · `prompts/market.md` · `prompts/stakeholder.md` · `prompts/rubrics/4.2` `4.3` |
| **C** | 민영은 | 도메인 · 종합 · 편향 장치 | `agents/domain.py` · `agents/synthesis.py` · `prompts/domain.md` · `prompts/synthesis.md` · `prompts/neutrality_judge.md` · `prompts/rubrics/4.1` `4.4` `4.5` |
| **D** | 황재원 | State · Graph · 보고서 · 발표 | `graph/` · `agents/report.py` · `app.py` · `prompts/report.md` · `outputs/report/` |

`config/` · `scripts/` · `pyproject.toml` · `.env.example` · CI · `README.md` 는 공용입니다. 바꾸기 전에 슬랙에 공유하세요.
누가 무엇을 기다리는지는 [docs/ROLES.md](docs/ROLES.md) 의존 그래프 참고.

**남의 파트에 변경이 필요하면 직접 고치지 말고 이슈를 만들어 소유자에게 넘깁니다.**

---

## 2. 브랜치 전략

개발이 하루라 `develop` 없이 **main + 작업 브랜치** 2단계로만 갑니다.

```
main ────●────────●────────●────────●──▶   (항상 python app.py 가 도는 상태)
          \      /  \     /  \     /
           feat/3   feat/7   fix/11        (이슈 1개 = 브랜치 1개)
```

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

> 이슈 페이지 오른쪽 **Development → Create a branch** 를 쓰면 브랜치가 이슈에 자동 연결되고,
> PR 머지 시 이슈가 자동으로 닫힙니다. GitHub 이 제안하는 이름(`3-feat-...`)은
> 그 팝업에서 `feat/3-stakeholder-worker` 로 **직접 고쳐서** 만드세요.

---

## 3. 작업 순서

```bash
# 1) 이슈 생성 (GitHub 웹에서 템플릿 선택) → 이슈 번호 확인 (#3)

# 2) 최신 main 받기
git switch main
git pull origin main

# 3) 브랜치 생성
git switch -c feat/3-stakeholder-worker

# 4) 작업 → 테스트 → 커밋
uv run pytest
git add .
git commit -m "feat(agents): 이해관계자 워커 — 찬반 각 2건 강제"

# 5) 푸시
git push -u origin feat/3-stakeholder-worker

# 6) GitHub에서 PR 생성 (템플릿 자동 삽입됨) → 리뷰 요청

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
- 본문의 `closes #3` 을 **반드시** 채우기 → 머지 시 이슈 자동 종료
- 리뷰어 **최소 1명** 승인 후 머지
- 머지 방식: **Squash and merge**
- 머지 후 원격 브랜치 삭제
- **스스로 머지하지 않기.** 리뷰가 급하면 슬랙에서 부르기

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
| **`main` 직접 push** | 아래 ⚠️ 참조 |

> ⚠️ **`main` 보호는 GitHub 이 강제하지 못할 수 있습니다.** 무료 플랜의 private 저장소에는
> branch protection·ruleset 을 걸 수 없습니다 (public 전환 시 가능).
> 서로 지키는 것으로 합니다.

---

## 7. 설계서와 코드가 어긋나면

설계서(`docs/설계서.md`)는 10시에 제출한 **정본**입니다. 개발하다 바꿔야 할 게 생기면:

1. `[CHORE] 설계 변경` 이슈로 올리고 조원 합의
2. 코드 PR 에서 "설계서 반영" 칸에 절 번호 적기
3. 바뀐 이유를 보고서 **6. 한계점** 또는 README **Lessons Learned** 에 한 줄

설계와 코드가 다른데 설명이 없으면 "설계 구현 충실도" 항목에서 감점됩니다.

---

## 8. 충돌이 났다면

```bash
git fetch origin && git rebase origin/main    # 내 브랜치를 최신 main 위로
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
