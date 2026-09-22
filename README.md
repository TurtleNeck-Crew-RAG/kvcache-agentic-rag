# KV cache 최적화 기술 다관점 평가 — Agentic RAG

본 프로젝트는 KV cache 최적화 기술을 SW(KIVI)·HW(InfiniGen) 두 진영에서 하나씩 선정하여,
**스마트폰 온디바이스 LLM** 도메인 안에서 기술 성숙도·시장·이해관계자·도메인 관점으로
비교 평가하는 Agentic RAG 를 개발하는 프로젝트임. 우열을 판정하지 않고, 관점에 따라 평가가 어떻게 엇갈리는지를 드러낸다.

> SKALA 판교 10반 · 박유진 · 황재원 · 민영은 · 심준용
> 설계서: [docs/설계서.md](docs/설계서.md) · 협업 규칙: [CONTRIBUTING.md](CONTRIBUTING.md) · 분담: [docs/ROLES.md](docs/ROLES.md)

## Overview
- Objective : 하나의 도메인(스마트폰 온디바이스 LLM)에서 두 기술을 4가지 관점으로 비교 평가
- Method : Multi-Agent — Branching(fan-out/fan-in) + Loop, 규칙 기반 Dispatcher + Agentic RAG
- Tools : LangGraph, Chroma, BGE-M3, Tavily, LangSmith, RAGAS

## Selected Technologies
- SW : **KIVI** (ICML 2024) — 사후·무보정 2bit KV 양자화. 재학습 없이 배포된 모델에 즉시 적용, 피크 메모리 2.6× 감소. 정확도 trade-off 가 관점 대조 재료 (설계서 1.3)
- HW : **InfiniGen** (OSDI 2024) — 동적 KV 오프로딩·프리패치. 전용 HW 없이 메모리 계층을 쓰는 유일한 후보, 정확도 무손실 vs 전송·전력 (설계서 1.4)
- 선정 방식 : 조가 기준표(온디바이스 적용 가능성 · 자료 확보 · 성숙도 · 코퍼스 적합성)로 6편 비교 후 선정 — `config/selection.yaml`

## Features
- 논문 2편(33p) 기반 사실 추출 — 페이지 인용 `[p.N]`, 기술별 독립 검색(`tech` 필터)
- 한국어 질의 → 영어 논문 cross-lingual 검색 — **이중 질의 하이브리드**: dense(BGE-M3) 는 한국어 원 질의, BM25 는 영어 번역 질의, RRF 0.5/0.5, k=4
- 관련성 체크 → 질문 재작성 1회 → 실패 시 "논문에 근거 없음" 기록 (재작성 전/후 로그)
- 시장·이해관계자는 Tavily 웹검색, 도메인은 RAG + 웹 (HW 우호 반례 필수 수집)
- 확증 편향 방지 전략 : 사전 가설 명시 · 기술별 독립 호출 · 사실 단위 질의 · 정확도 임계값 없음 · 출처 태그 강제 `[논문]/[웹]/[추론]` · 반대 근거 ≥2 강제(≤2회 재검색) · 중립성 검증 루프(≤2회) — 설계서 5.5
- 보고서 REFERENCE 는 실제 인용된 `citations` 에서만 자동 생성

## Tech Stack
- Framework : LangGraph 1.x (Python 3.11, uv)
- LLM/Generator : gpt-4.1-mini
- LLM/Judge : gpt-4.1-mini (temperature 0, Generator 와 별도 인스턴스) · 경량 변환 gpt-4.1-nano
- Retrieval : Chroma(dense) + BM25(sparse, 영어 번역 질의) RRF 0.5/0.5, k=4 — **Hit Rate@4 0.80, MRR@4 0.52** (20문항; dense 단독 0.55/0.40, 설계 초기값 M3-sparse 하이브리드 0.45 → 실측으로 BM25 확정: [experiments/sparse_compare](experiments/sparse_compare/README.md))
- Embedding : `BAAI/bge-m3` (오픈소스, 로컬) — 후보 4개 대조군 실측, 한국어 질의·max_seq 8192 요건으로 선정 (설계서 3.5)
- Generation eval : RAGAS **Faithfulness 0.936 · ResponseRelevancy 0.838 · LLMContextPrecisionWithoutReference 0.912** (20문항, 근거 없음 0)
- Observability : LangSmith 프로젝트 `kv-cache-eval`

## Agents

| 에이전트 | 역할 | RAG | 도구 | 출력 키 |
|---|---|---|---|---|
| Dispatcher | 워커 호출·재호출을 State 규칙으로 결정. LLM 없음 | — | — | `next`, `retry` |
| 기술 조사 | 논문에서 개요·메커니즘·수치·한계·적용조건 추출 (기술별 독립) | O | retriever | `tech_summary` |
| 시장 평가 | 시장 규모·채택·생태계 | X | Tavily | `market_eval` |
| 이해관계자 평가 | 경쟁 진영·개발자·투자 반응, 찬반 각 ≥2 | X | Tavily | `stakeholder_eval` |
| 도메인 평가 | 논문 사실 추출 → 온디바이스 제약으로 판정, HW 반례 웹검색 | O | retriever + Tavily | `domain_eval` |
| 평가 종합 | 관점 × 기술 매트릭스, 일치/상충, TRL, 중립성 검증 | X | — | `synthesis`, `trl_estimate`, `neutrality` |
| 보고서 생성 | 6장 목차 md → PDF, REFERENCE 자동 | X | — | `report_md` |

## Architecture

![graph](docs/images/graph.png)

설계 그림(위)과 `python -m graph.build` 가 그린 컴파일 결과([graph_compiled.png](docs/images/graph_compiled.png))가 같다 — 워커 6개가 전부 `dispatcher` 로 수렴하고, 조건부 엣지가 7갈래(+END)로 나간다.

인덱싱 → ① 기술 조사 → ② 평가 3개 병렬(fan-out, 분리 키) → ③ 종합 + 중립성 Judge → ④ 보고서.
Loop: 관련성 재작성 ≤1(RAG 노드 내부) · 반대 근거 부족 ≤2 · 중립성 반려 ≤2. `llm_calls > 100` 이면 재호출 중단.

RAG 파이프라인: [전처리](docs/images/pipeline_pre.png) · [검색·관련성](docs/images/pipeline_ret.png) · [생성·사실성](docs/images/pipeline_gen.png)

## Directory Structure

```
├── app.py                 # 실행 스크립트 — 인덱싱 확인 → 그래프 실행 → outputs/
├── graph/                 # state.py(State 스키마) · dispatcher.py(규칙표) · build.py(조립)
├── agents/                # 워커 6개 — run(state) -> dict
├── rag/                   # indexing · retriever · judge · rag_node(ask) · evaluate
├── prompts/               # 에이전트별 프롬프트 .md + rubrics/(4.1~4.5)
├── config/                # domain.yaml(0.3 환경) · selection.yaml(1장 선정)
├── data/                  # papers/(논문 PDF, git 제외) · index/ · cache/
├── experiments/           # 임베딩 실측(embed_compare) · sparse 비교
├── outputs/               # report/(보고서 md·PDF) · retrieval_log.json · citations.json
├── tests/                 # LLM 없이 도는 단위 테스트 (dispatcher 규칙표)
├── scripts/               # fetch_papers.sh
├── docs/                  # 설계서.md · ROLES.md · images/
└── README.md
```

## Usage

```bash
uv sync --extra pdf              # Python 3.11, .venv — md→PDF 는 weasyprint(brew install pango)
cp .env.example .env             # OPENAI_API_KEY · TAVILY_API_KEY · LANGSMITH_API_KEY
bash scripts/fetch_papers.sh     # arXiv 2402.02750, 2406.19707 → data/papers/
uv run python -m rag.indexing    # BGE-M3 임베딩(첫 실행 시 모델 다운로드) → data/index/
uv run python app.py             # 그래프 실행 → outputs/report/report.md (+ .pdf)
#   --skip-index                    인덱싱 건너뜀 / --pdf-name <파일명>  제출용 PDF 이름
#   실패해도 outputs/run.json (visited · llm_calls · retry · error) 과 채워진 State 키의 .json 은 남는다
uv run python -m agents.report   # 보고서 워커만 fixtures 로 단독 실행 (LLM 4회)

uv run python -m rag.evaluate    # Hit Rate@4 · MRR@4 · RAGAS
uv run pytest                    # 단위 테스트
```

## Contributors
- 박유진 : RAG Pipeline (Indexing · Hybrid Retrieval · Relevance/Faithfulness Judge), Embedding 실측·선정, 기술 조사 Agent, Repo 구성
- 심준용 : 기술 선정 기준표·HW 선정 사유, 시장 평가 Agent, 이해관계자 평가 Agent (반대 근거 강제)
- 민영은 : 문제 정의·도메인 설정, 도메인 평가 Agent, 평가 종합 Agent, 확증편향 방지 장치·중립성 Judge
- 황재원 : State Schema · Dispatcher · Graph 설계/구현, 보고서 생성 Agent, md→PDF, README 취합·발표

## Lessons Learned
<!-- 발표 말미용. 개발 끝나고 채운다 — 설계와 달라진 것, 실측이 가설을 뒤집은 것, 다음에 다르게 할 것 -->
- 설계서 3.4 의 sparse 초기값(BGE-M3 learned sparse)은 실측에서 뒤집혔다 — 한국어 질의에 sparse 를 섞으면 dense 단독보다 나빠지고(0.55→0.45), 해법은 sparse 모델 교체가 아니라 **질의 언어 분리**(dense ← 한국어, BM25 ← 영어 번역)였다. 리더보드가 아니라 우리 질의로 재야 보이는 것 ([실측](experiments/sparse_compare/README.md))
- TBD
