# tests/fixtures/

워커 개발을 서로 기다리지 않기 위한 **손으로 쓴 State 예시**. 형식은 `graph/state.py` 와 같다.

| 파일 | 용도 | 쓰는 사람 |
|---|---|---|
| `tech_summary.json` | `state["tech_summary"]` 예시 (KIVI · InfiniGen) | B market/stakeholder · C domain 입력 |
| `evals.json` | `market_eval` · `stakeholder_eval` · `domain_eval` 예시 | C synthesis · D report 입력 |

```python
import json
from graph.state import init_state
s = init_state({}, {})
s["tech_summary"] = json.load(open("tests/fixtures/tech_summary.json"))
s["tech_summary"].pop("_note")
out = my_worker.run(s)
```

내용은 placeholder 다 — 실제 값은 워커가 만든다. 진짜 데이터가 나오면 이 파일을 실제 출력으로 교체해도 된다.
