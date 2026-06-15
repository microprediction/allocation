# When Does Portfolio Construction Work?

- **Status**: draft
- **Authors**: Peter Cotton
- **Slug**: `online-portfolio-regimes`

## Abstract

A portfolio rule is a covariance estimate feeding an allocator. Which rules work
out of sample — after trading costs, and as the number of assets grows? We
backtest eleven online constructions over thousands of random subsets of U.S.
stocks and map the winner against two axes: the number of names and the cost of
trading. The map has three regions — dense minimum-variance (few names, cheap
trading), factor minimum-variance (many names, cheap trading; the dense version
is undefined there), and inverse-variance (any size, once costs exceed ~5–10 bp,
because turnover binds). We give the two fixes that make the many-names corner
work (Woodbury inversion of a factor covariance; a ridge-regularized Schur
coupling) and a diagnostic showing the signed methods' short positions are not
needed on this universe.

## Files

- `paper.tex` — manuscript (imports `../shared/preamble.tex`, cites `../refs.bib`)
- `figures/` — figures specific to this paper

## Build

```sh
cd .. && ./build.sh online-portfolio-regimes   # -> paper.pdf (requires tectonic)
```

## Reproducing the numbers

The regime map, γ-crossover, and short-diagnostic figures come from the
`allocation.backtest` Monte-Carlo harness over the `winningportstudy` panel; no
data is vendored. See `allocation/diagnostics.py` for the negative-weight test.
