#!/usr/bin/env bash
# Render each paper's LaTeX to an HTML reading page under docs/papers/<slug>/.
# Requires pandoc. Run from anywhere.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"          # papers/
render() {                                      # $1 = slug
  local slug="$1" out="$here/../docs/papers/$1"
  mkdir -p "$out/figures"
  cp "$here/$slug/figures/"*.png "$out/figures/" 2>/dev/null || true
  ( cd "$here/$slug" && pandoc paper.tex -f latex -t html5 --standalone --mathjax \
      --citeproc --bibliography=../refs.bib \
      --metadata title="$2" --template=../web-template.html -o "$out/index.html" )
  echo "wrote docs/papers/$slug/index.html"
}
render thurstone-portfolios "Thurstone Portfolios: Allocation as Winning Probability"
