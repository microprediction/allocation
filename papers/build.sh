#!/usr/bin/env bash
# Build Thurstone papers with Tectonic.
#
# Usage:
#   ./build.sh              # build every paper (each <slug>/paper.tex)
#   ./build.sh <slug>       # build just papers/<slug>/paper.tex
#
# Output PDFs are written next to each paper.tex. The shared bibliography
# (papers/refs.bib) and preamble (papers/shared/preamble.tex) are resolved
# via the relative paths in each paper. Tectonic runs biber automatically.

set -euo pipefail

cd "$(dirname "$0")"

if ! command -v tectonic >/dev/null 2>&1; then
  echo "error: tectonic not found. Install with: brew install tectonic" >&2
  exit 1
fi

build_one() {
  local tex="$1"
  echo ">> building $tex"
  # Run from the paper's own directory so ../refs.bib, ../shared, and figures/
  # resolve correctly.
  ( cd "$(dirname "$tex")" && tectonic --synctex --keep-logs "$(basename "$tex")" )
}

if [[ $# -ge 1 ]]; then
  tex="$1/paper.tex"
  [[ -f "$tex" ]] || { echo "error: no such paper: $tex" >&2; exit 1; }
  build_one "$tex"
else
  shopt -s nullglob
  found=0
  for tex in */paper.tex; do
    [[ "$tex" == _template/* ]] && continue
    build_one "$tex"
    found=1
  done
  [[ "$found" -eq 1 ]] || echo "no papers found yet (only _template/). Copy _template/ to <slug>/ to start one."
fi
