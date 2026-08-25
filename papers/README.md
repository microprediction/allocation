# Papers

Working papers and write-ups for `allocation`.
Papers are written in LaTeX and compiled with [Tectonic](https://tectonic-typesetting.github.io/).

All papers share one bibliography (`refs.bib`) and one preamble
(`shared/preamble.tex`), so citations and styling stay consistent.

```
papers/
  refs.bib           # SHARED bibliography — all papers cite from here
  shared/
    preamble.tex     # SHARED preamble (packages, biblatex setup, macros)
  build.sh           # compile papers with tectonic
  <short-paper-slug>/
    paper.tex        # the manuscript (imports ../shared/preamble.tex)
    README.md        # title, status, abstract, authors
    figures/         # figures specific to this paper
  _template/         # copy this to <slug>/ to start a new paper
```

## Papers

| Slug | Title | Status |
|------|-------|--------|
| [thurstone-portfolios](thurstone-portfolios/) | Thurstone Portfolios: Long-Only Allocation by Inverting Winning Probabilities | draft |
| [repaired-hierarchical-portfolios](repaired-hierarchical-portfolios/) | Repaired Hierarchical Portfolios | draft |
| [thurstone-credit](thurstone-credit/) | Winning Probabilities as Credit: Fast, Redundancy-Aware Attribution | draft |
| [online-portfolio-regimes](online-portfolio-regimes/) | When Does Portfolio Construction Work? A Map across the Number of Assets and the Cost of Trading | draft |

### Portfolio papers kept in other repos

Not every portfolio-construction paper lives here. The Schur-complementary
family is split off in [`microprediction/schur`](https://github.com/microprediction/schur),
`paper/`:

- `neither-end-of-the-bridge.tex` — When the Out-of-Sample-Optimal Schur
  Portfolio Lies Between HRP and Minimum Variance
- `schur-damping-pdlp.tex` — Schur Damping for Perpetual Demand Lending Pools

Both build directly on the original Schur-complementary allocation paper
(`cotton2024schur`, arXiv:2411.05807), which this repo's papers also cite.
This split is historical, not principled; check both locations before
assuming you've found every portfolio paper.

## Starting a new paper

```sh
cp -r _template special-horse-calibration   # pick a kebab-case slug
# edit special-horse-calibration/paper.tex (title, authors, content)
./build.sh special-horse-calibration        # -> special-horse-calibration/paper.pdf
```

## Building

```sh
./build.sh            # build every paper
./build.sh <slug>     # build one paper
```

Requires `tectonic` (`brew install tectonic`). Tectonic fetches packages on
first run and runs bibtex for the bibliography automatically. (We use biblatex's
`backend=bibtex`, which Tectonic runs natively; the system `biber` is not used.)

## Conventions

- **Slug**: short, kebab-case, stable (e.g. `special-horse-calibration`).
- **Status**: `draft` → `internal-review` → `submitted` → `published`.
- **Bibliography**: add entries to the shared `refs.bib`; don't keep per-paper
  bib files. Cite with `\textcite{key}` / `\parencite{key}`.
