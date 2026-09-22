"""실행 스크립트 — python app.py

1. config/domain.yaml · config/selection.yaml 로드 → init_state
2. 인덱스가 없으면 rag.indexing.build_index()
3. graph 실행 (recursion_limit 40 — 안전망)
4. outputs/report/ 에 md 저장 → PDF 변환, retrieval_log · citations 도 outputs/ 에 json 으로
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

from graph.build import build_graph
from graph.state import init_state

OUT = Path("outputs")


def main() -> None:
    load_dotenv()
    domain = yaml.safe_load(Path("config/domain.yaml").read_text())
    selected = yaml.safe_load(Path("config/selection.yaml").read_text())

    app = build_graph()
    final = app.invoke(init_state(domain, selected), config={"recursion_limit": 40})

    (OUT / "report").mkdir(parents=True, exist_ok=True)
    (OUT / "report" / "report.md").write_text(final["report_md"])
    for key in ("retrieval_log", "citations", "synthesis", "trl_estimate"):
        (OUT / f"{key}.json").write_text(json.dumps(final.get(key), ensure_ascii=False, indent=2))
    print(f"llm_calls={final.get('llm_calls')}  retry={final.get('retry')}  → outputs/report/report.md")


if __name__ == "__main__":
    main()
