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
  publish "$(dirname "$tex")"
}

# Papers are LaTeX and ship as PDFs. If a paper has a docs/ directory it is published on the
# site, so copy the freshly built PDF there and leave index.html as a redirect, which keeps
# older inbound links (.../<slug>/index.html) working instead of 404ing.
publish() {
  local slug="$1" dest="../docs/papers/$1"
  [[ -d "$dest" ]] || return 0
  cp "$slug/paper.pdf" "$dest/paper.pdf"
  cat > "$dest/index.html" <<HTML
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>$slug</title>
<link rel="canonical" href="paper.pdf">
<meta http-equiv="refresh" content="0; url=paper.pdf">
</head><body>
<p>This paper is a PDF built from LaTeX: <a href="paper.pdf">paper.pdf</a>.</p>
</body></html>
HTML
  echo "   published -> docs/papers/$slug/paper.pdf"
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
