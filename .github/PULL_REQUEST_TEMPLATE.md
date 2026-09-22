<!-- 제목: [FEAT] 이해관계자 워커 — 찬반 각 2건 강제  (이슈 제목과 맞추기) -->

## 관련 이슈
- closes #

> 예시: closes #3

---

## 변경 사항
이번 PR에서 어떤 작업을 했는지 간단하게 작성해주세요.

- 
- 
- 

---

## 건드린 디렉토리
수정한 곳에 체크해주세요. **자기 파트만 체크되어야 정상입니다.**

- [ ] `rag/` &nbsp;/&nbsp; `agents/tech_research.py` &nbsp;/&nbsp; `experiments/` &nbsp;/&nbsp; `prompts/rag_*.md` `tech_research.md` — **A** 박유진
- [ ] `agents/market.py` &nbsp;/&nbsp; `agents/stakeholder.py` &nbsp;/&nbsp; `prompts/market.md` `stakeholder.md` &nbsp;/&nbsp; `prompts/rubrics/4.2` `4.3` — **B** 심준용
- [ ] `agents/domain.py` &nbsp;/&nbsp; `agents/synthesis.py` &nbsp;/&nbsp; `prompts/domain.md` `synthesis.md` `neutrality_judge.md` &nbsp;/&nbsp; `prompts/rubrics/4.1` `4.4` `4.5` — **C** 민영은
- [ ] `graph/` &nbsp;/&nbsp; `agents/report.py` &nbsp;/&nbsp; `app.py` &nbsp;/&nbsp; `prompts/report.md` — **D** 황재원
- [ ] `graph/state.py` ← 체크되면 **키 이름·타입이 바뀐 것이니 슬랙에 알리세요** (모든 워커가 의존)
- [ ] `agents/_common.py` &nbsp;/&nbsp; `config/` &nbsp;/&nbsp; `pyproject.toml` &nbsp;/&nbsp; `.env.example` &nbsp;/&nbsp; CI (공용)
- [ ] `tests/fixtures/` (예시 State — 형식이 바뀌면 함께 갱신)
- [ ] 문서만 수정

---

## 동작 확인
- [ ] `uv run pytest` 통과
- [ ] 워커 단독 실행 (`tests/fixtures/` 입력) → 출력 키가 `graph/state.py` 타입과 맞음
- [ ] `uv run python app.py` 끝까지 통과 → LangSmith 링크: 
- [ ] 끝까지 안 됨 → 어디까지 되는지:

<!-- 에러 요약 / llm_calls 수 / retry 카운터 -->

---

## 설계서 반영
- [ ] `docs/설계서.md` 수정이 필요 없습니다
- [ ] 필요합니다 → 어느 절인지: <!-- 예: 3.4 Retriever 행 — sparse 를 BM25 로 확정 --> 이유는 보고서 6장 한계점 / README Lessons Learned 에도 한 줄

---

## 출력 예시 (선택)
워커가 반환한 State 키(JSON)나 보고서 일부를 붙여주세요. (README · 발표에 그대로 씁니다)

```json
```

---

## 리뷰어가 볼 것
<!-- 리뷰어가 30초 안에 판단할 수 있게 — "프롬프트 wording 만 봐줘" / "retry 로직이 규칙 2' 과 맞는지" -->
- 

---

## 체크리스트
- [ ] 브랜치 네이밍 규칙을 준수했습니다. (`feat/이슈번호-slug`)
- [ ] 관련 이슈를 연결했습니다. (`closes #이슈번호`)
- [ ] 커밋 메시지가 `<type>(<scope>): 요약` 형식입니다.
- [ ] API 키를 코드에 직접 적지 않았습니다. (`.env` + `load_dotenv`)
- [ ] 프롬프트는 `prompts/*.md` 에 두고 `load_prompt()` 로 읽습니다. (코드 안 긴 문자열 X)
- [ ] 두 기술을 같은 프롬프트에 넣지 않았습니다. (`for tech in TECHS`)
- [ ] 판정 문장에 `[논문 p.N]` / `[웹 URL]` / `[추론]` 태그가 있습니다.
- [ ] 우열·추천 표현("더 낫다", "권장", "선택해야")을 쓰지 않았습니다.
- [ ] 워커는 자기 출력 키 + `citations` + `llm_calls`(+`retrieval_log`) 만 반환합니다.
- [ ] `data/papers/*.pdf`, `data/index/`, `.env` 를 커밋하지 않았습니다.
- [ ] 남의 디렉토리를 고치지 않았습니다.
