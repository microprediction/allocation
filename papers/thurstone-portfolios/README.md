# Thurstone Portfolios: Long-Only Allocation by Inverting Winning Probabilities

- **Status**: draft
- **Authors**: Peter Cotton
- **Started**: 2026-06-04

## Abstract

A long-only asset-allocation scheme in which portfolio weights are the winning
probabilities of a Thurstonian race among assets. Starting from a variance-only
(diagonal) allocation, weights are inverted into latent abilities, the contest is
re-run with correlation reintroduced, and each asset's probability of "winning"
becomes its new weight. The construction is long-only and budget-feasible by
design, avoids covariance inversion, and exposes a single correlation dial $\phi$
in place of mean--variance shrinkage. Positioned as a non-Lucian alternative to
CAPM / capitalization weighting.

## Status / TODO

This is an early draft. Method, CAPM framing, and protocol are written; empirical
results are placeholders.

- [ ] Fill REIT study results (from `winningportstudy`).
- [ ] Fill Russell / AI-benefit study results (from `winningetfstudy`).
- [ ] Deterministic correlated-race (lattice) implementation in `thurstone`.
- [ ] Figures: efficient-frontier comparison, $\phi$ sensitivity, drawdown curves.

## Sources

- Method/code: [`winningport`](https://github.com/microprediction/winningport)
  (`abilityport` / `ability_port_factory`), the legacy "winning"-package
  implementation of the ability tilt.
- Studies: [`winningportstudy`](https://github.com/microprediction/winningportstudy)
  (REITs), [`winningetfstudy`](https://github.com/microprediction/winningetfstudy)
  (Russell indices, AI-benefit scores).
- Narrative: `docs/book/chapters/{02-introduction,10-portfolios}` in this repo.

## Files

- `paper.tex` — manuscript (imports `../shared/preamble.tex`, cites `../refs.bib`)
- `figures/` — figures used in this paper
- Build: `../build.sh thurstone-portfolios` → `paper.pdf`
