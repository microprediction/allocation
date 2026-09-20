#!/usr/bin/env bash
# Render each paper's LaTeX to an HTML reading page under docs/papers/<slug>/.
# Requires pandoc. Run from anywhere.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"          # papers/
render() {                                      # $1 = slug
  local slug="$1" out="$here/../docs/papers/$1"
  mkdir -p "$out/figures"
  cp "$here/$slug/figures/"*.png "$out/figures/" 2>/dev/null || true
  # --shift-heading-level-by=1: pandoc renders \section as <h1>, which collides with the template's title <h1>
  # (the page had twelve of them) and inverted the visual hierarchy, since the stylesheet reserves <h1> for the
  # title. Sections become <h2>, subsections <h3>. --toc gives an 11k-word paper a contents block.
  ( cd "$here/$slug" && pandoc paper.tex -f latex -t html5 --standalone --mathjax \
      --citeproc --number-sections --shift-heading-level-by=1 --toc --toc-depth=2 \
      --bibliography=../refs.bib \
      --metadata title="$2" --template=../web-template.html -o "$out/index.html" )
  # repair what pandoc gets wrong for this LaTeX: the algpseudocode block and
  # multi-label \Cref{a,b} leaks (single LaTeX source stays authoritative for PDF).
  python3 "$here/_postprocess.py" "$out/index.html"
  echo "wrote docs/papers/$slug/index.html"
}
render thurstone-portfolios "Thurstone Portfolio Polishing: Tail-Sensitive Black-Litterman and Beyond"
render thurstone-credit "Winning Probabilities as Credit: Fast, Redundancy-Aware Attribution"
render online-portfolio-regimes "When Does Portfolio Construction Work? A Map across the Number of Assets and the Cost of Trading"
