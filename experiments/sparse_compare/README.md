# 검색 실측 — sparse 후보 · 하이브리드 구성 · RAGAS

2026-09-22 · 로컬 macOS arm64 (MPS) · 실행자: 유진 · 재실행 `uv run python -m rag.evaluate` → `outputs/eval.json` (이 폴더에 복사)

## 목적

설계서 3.4 가 "실측 후 확정"으로 남긴 것: ① sparse 후보 **BGE-M3 learned sparse vs BM25** ② 하이브리드가 dense 단독 0.65 → **Hit@4 ≥ 0.80** 을 달성하는가. 덤으로 ③ RAGAS 생성 품질, ④ 질의 재작성 효과.

## 설정

| 항목 | 값 |
|---|---|
| 코퍼스 | `data/papers/kivi.pdf`(15p) + `infinigen.pdf`(18p), `rag.indexing` 청킹 600/100 → **293 chunks** (embed_compare 는 페이지 슬라이딩 299) |
| 평가셋 | `../embed_compare/eval_set.json` 20문항 (KIVI 10 · InfiniGen 10), ko/en 쌍 + gold 문구 |
| 정답 판정 | top-4 청크 중 gold 문구(공백·대소문자 무시) 포함 → Hit |
| dense | BGE-M3 (Chroma, `tech` 필터) |
| sparse | `m3` = BGE-M3 lexical_weights 내적 / `bm25` = rank_bm25 (소문자·영숫자 토큰) |
| 융합 | RRF c=60, 가중 0.5/0.5, 후보 각 8 → top 4 |
| 번역 | `dual-*` 는 ko 질의를 gpt-4.1-mini 로 영어 검색 질의로 변환 (`translations.json`) |

## 결과 — 단일 질의 (질의 언어별)

| 모드 | ko Hit@4 | ko MRR@4 | en Hit@4 | en MRR@4 |
|---|---|---|---|---|
| dense | **0.55** | 0.40 | 0.50 | 0.38 |
| sparse-m3 | 0.45 | 0.25 | 0.70 | **0.53** |
| sparse-bm25 | 0.30 | 0.17 | **0.80** | 0.49 |
| hybrid-m3 (설계 초기값) | 0.45 | 0.32 | 0.65 | 0.47 |
| hybrid-bm25 | 0.40 | 0.31 | 0.70 | 0.50 |

## 결과 — 이중 질의 (dense ← ko 원 질의, sparse ← 영어 질의)

| 구성 | Hit@4 | MRR@4 | miss |
|---|---|---|---|
| dense-ko + bm25(사람이 쓴 en) 0.5/0.5 | 0.80 | 0.47 | 8, 10, 14, 16 |
| dense-ko + bm25(**nano** 번역) 0.5/0.5 | 0.75 | 0.50 | 3, 8, 10, 12, 14 |
| **dense-ko + bm25(mini 번역) 0.5/0.5 — 채택** | **0.80** | **0.52** | 3, 8, 10, 14 |
| dense-ko + bm25(mini 번역) 0.3/0.7 | 0.80 | 0.52 | 3, 8, 10, 14 |
| dense-ko + m3(mini 번역) 0.5/0.5 | 0.65 | 0.50 | 8, 9, 12, 14–17 |

## 해석

1. **한국어 질의에 sparse 를 섞으면 dense 보다 나빠진다** (0.55 → 0.45/0.40). 한국어 질의엔 영문 토큰이 없어 sparse 리스트가 잡음이고, RRF 가 dense 순위를 희석. 설계서 3.5 해석 4 의 우려("한국어 질의에 해당 영문 토큰이 없어 BM25 가 안 걸림")가 M3 sparse 에도 그대로 적용됨 — learned sparse 도 질의에 없는 토큰을 만들어내진 못함.
2. **영어 질의에는 BM25 > M3 sparse** (0.80 vs 0.70). M3 lexical_weights 에는 IDF 가 없어 `InfiniGen`·`GPU`·`KV` 같은 빈출 토큰이 점수를 지배 → 고유명사(A6000)·수치 매칭에서 BM25 가 우세. 설계서 3.4 의 초기값(M3 sparse) **뒤집힘**.
3. **해법 = 질의 언어 분리**: dense 는 cross-lingual 이 되는 BGE-M3 에 한국어 원 질의, BM25 에는 영어 번역 질의. 0.55 → **0.80**, 목표 달성. dense 만 영어로 바꾸면 오히려 하락(dense/en 0.50)이라 dense 는 한국어 유지.
4. **번역 모델**: nano 0.75 / mini 0.80 — nano 는 "KIVI peak memory reduction amount" 처럼 정보량은 유지하나 고유명사·수치를 빠뜨리는 경우가 있어 mini 채택. 설계서 3.6 은 "판단 없는 변환 = nano" 였으나 검색 성패를 가르므로 재작성(mini)과 같은 등급으로 — 호출 +1/질의 (ask 당 3 → 4).
5. 남은 miss 4 (#3 KIVI 파인튜닝 불필요, #8 per-channel 이유, #10·#14 InfiniGen) 는 gold 가 본문 문장 속 짧은 구절이라 청크 경계·표 분할 문제 — n=20 에서 1문항 = 0.05.

## 질의 재작성 효과 (retrieval_log)

| 시점 | 질의 수 | 재작성 | 구제(yes) | 근거 없음 |
|---|---|---|---|---|
| 이중 질의 도입 전 (hybrid-m3) | 11 (대체 질문 1 포함) | 2 | 1 | 1 |
| 이중 질의 도입 후 (dual-bm25) | 10 | 3 | **3** | 0 |

도입 후 기술 조사 10문항 전부 근거 확보. 재작성이 3건 모두 구제 → 유지 (보고서 한계점 6). 재작성 질의는 영어로 나오므로 dense·BM25 양쪽에 그대로 사용.

## RAGAS — 생성 품질 (20문항 ko, 이중 질의 하이브리드, gpt-4.1-mini judge)

| Faithfulness | ResponseRelevancy | LLMContextPrecisionWithoutReference | 근거 없음 | ask llm_calls |
|---|---|---|---|---|
| **0.936** | **0.838** | **0.912** | 0 / 20 | 80 (4/문항) |

## 설계서와 달라진 것 (보고서 한계점 5 · README Lessons Learned 에 기록)

- 3.4 Retriever 행: "BGE-M3 sparse 초기값, BM25 와 비교 후 확정" → **BM25 확정**, 단 **영어 번역 질의**에 적용
- 3.6: 번역 단계 추가 (mini). 기술 조사 1회 실행 35 → **46 calls** (실측)
- 3.5 해석 5 의 대안(chunk 1000 · parent-document · 요약검색)은 목표 달성으로 미측정
