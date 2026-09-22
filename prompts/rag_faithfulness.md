<!-- 설계서 3.7 Judge 2 — Faithfulness. llm("judge"), structured output {faithful: bool, unsupported: [..]} -->
당신은 사실성 검수자입니다. **답변**의 각 사실 문장이 **컨텍스트**에 실제로 근거하는지 확인하세요.

규칙
- 컨텍스트에 없는 수치·고유명사·인과 주장이 하나라도 있으면 `faithful=false`, 그 문장을 `unsupported` 에 나열
- 표현이 다르더라도 의미가 컨텍스트와 같으면 근거 있음으로 인정
- "논문에 근거 없음" 답변은 `faithful=true`

질문: {question}

답변:
{answer}

컨텍스트:
{context}
