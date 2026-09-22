#!/usr/bin/env bash
# 논문 PDF 는 커밋하지 않는다 (설계서 7.2). 클론 직후 이 스크립트로 받아야 인덱싱이 재현됨.
#   bash scripts/fetch_papers.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/papers

fetch() {  # $1 = arXiv id, $2 = 저장 이름
  if [ -f "data/papers/$2.pdf" ]; then echo "skip  $2.pdf (있음)"; return; fi
  echo "fetch $2.pdf  <- arXiv:$1"
  curl -fsSL "https://arxiv.org/pdf/$1" -o "data/papers/$2.pdf"
}

fetch 2402.02750 kivi        # KIVI (ICML 2024)  15p
fetch 2406.19707 infinigen   # InfiniGen (OSDI 2024) 18p

python - <<'PY'
import fitz, pathlib
for p in sorted(pathlib.Path("data/papers").glob("*.pdf")):
    print(f"{p.name}: {fitz.open(p).page_count}p")
PY
