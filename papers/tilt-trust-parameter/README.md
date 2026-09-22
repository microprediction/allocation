# A Tilt Is Worth What the Correlation Is Worth

- **Status**: draft
- **Authors**: Peter Cotton
- **Started**: 2026-08-25 (merged with the tilt-trust-parameter draft 2026-09-22)

## Abstract

Racing a benchmark under a different dependence law is a diversification
operator. Calibrate the latent abilities that reproduce a long-only portfolio
under one correlation, re-evaluate them under another, and the weights move
toward whatever the first law was missing. Nothing is inverted.

Two things govern what it is worth and they are independent. The race halves
the variance of equal weight and of inverse variance and moves a long-only
minimum-variance portfolio by a tenth of a percent, so what is available is
bounded by how much correlation the benchmark already holds. And the damping
is a trust parameter: a full tilt costs nineteen percent of variance at a
tenth of an observation per asset and saves thirteen at five, with the optimum
walking monotonically between.

Hierarchical portfolios are the leading application, because recursive
bisection discards cross-block covariance by construction. On a known-truth
contagious market the tilt cuts out-of-sample ES95 by 9-17% against plain HRP
and Schur. Across 12,000 real S&P 500 and REIT trials it wins 59-80% under
every covariance estimator tried, and the largest tilt, calibrated against an
independent reference, is the single best setting.

Two negative results: the tail-aware Student-t race is consistently worse than
the Gaussian one even on a tail metric, and the operator must be applied once
rather than iterated. A confidence-weighted tail view is bounded-regret
regardless of whether the belief is correct.

## History

This paper is the merge of two drafts that turned out to be the same
construction approached from opposite ends. `repaired-hierarchical-portfolios`
fixed the race law at the full estimate and moved the *reference* from
independence toward the tree, indexed by `F`. `tilt-trust-parameter` fixed the
reference at independence and moved the *race law* toward the estimate,
indexed by `phi`. Both interpolate between leaving the benchmark alone and
racing it fully under the estimate, and the repair operator's fixed point
(`C_used = C_full`) is the `phi = 0` identity. The general frame is kept, the
HRP/Schur material is the leading application, and the real-market evidence
comes from the repair draft.

## Status / TODO

Early draft. Numbers in the paper are real (not placeholders): the synthetic
(bowling-lab) tables are produced by the three scripts below at reduced
quadrature/Monte-Carlo resolution for exploratory speed; the real-market table is
from a 12,000-trial, ~6.4-hour overnight run with zero errors
(`experiments/overnight_repair_validation.py`, `experiments/data/overnight_results.csv`,
not committed -- see `.gitignore`).

- [ ] Re-run key synthetic tables at production resolution (full points, Q, Sobol budget).
- [x] Build and test the gamma-blended reference `(1-gamma)*C_tree + gamma*Sigma_hat`
      for Schur -- done, Section 4.6 (`experiments/gamma_blend_check.py`): fixes F=0's
      high-gamma failure, trades away some low-gamma edge; an adaptive F=0-or-blend
      policy (switching on gamma alone) beats either fixed policy 4-6x in mean gain.
- [ ] Try the other alternative `C_used` construction floated in the discussion
      section: a flat K-cluster block-diagonal reference with exact (not
      factor-approximated) within-cluster covariance.
- [ ] Sharper tail-dependence test: ES99 or joint-crash probability instead of ES95,
      larger in-sample window for the rare cascade rows, tighter cluster geometry;
      and, on real data, dig into WHY REIT sub-portfolios reward the historical-bootstrap
      repair while S&P 500 sub-portfolios don't (sector concentration? a common
      rate factor? sub-portfolio size alone doesn't fully explain it).
- [x] Real-market validation (not just the bowling generator) -- done, Section 4.5.
- [x] Check the result isn't an artifact of this package's Fiedler seriation vs.
      classical (Lopez de Prado) single-linkage HRP -- done, Section 4.5, holds under
      both (`experiments/seriation_check.py`).
- [x] Decompose the gain into mean return / volatility / Sharpe / ES95, not ES95
      alone -- done, Section 4.5 (`experiments/return_metrics_check.py`): it's a
      volatility/tail-risk effect, not a return forecast.
- [x] Check whether the repair is safe to iterate -- done, Discussion ("Apply once;
      do not iterate", `experiments/iterate_repair_check.py`): it is not a fixed
      point of itself and should not be re-applied to its own output.
- [x] A confidence-weighted tail view (Black-Litterman-style, given a belief rather
      than trying to estimate one) -- done, Section 4.7 (`experiments/
      tail_confidence_dial_check.py`): correct belief helps monotonically, wrong
      belief costs boundedly (comparable magnitude, not catastrophic), an imprecise
      belief (wrong nu, right qualitative direction) still helps.
- [ ] Figures: ES95 vs. max_factors curves; gain vs. T_in; win rate by estimator.

### Open threads NOT yet in the paper (real findings, need more work before write-up)

A later, broader "which allocators does the repair help" investigation (not yet
reflected above) found a major methodological gap worth flagging before any of it
goes in: every comparison in this paper (and everything below) uses a ONE-SHOT,
cold-start, fresh-Monte-Carlo-seed protocol -- not the common-seed-transport /
streaming machinery (`BaseOnlinePortfolio.partial_fit`) the smoothness theorem is
actually about. The one result re-checked under a fair walk-forward protocol
(`experiments/turnover_walkforward_check.py`) saw its performance edge over simple
alternatives largely evaporate while its turnover cost (2-3x higher) did not --
i.e. the one-shot protocol can meaningfully overstate the repair's case. None of
the following should be treated as confirmed until re-verified the same way:

- The repair helps allocators in proportion to how little correlation they already
  use (naive/blind: inverse-variance, HRP, equal-weight all win; already-optimized:
  min-variance, max-decorrelation lose) -- `experiments/naive_baseline_repair_check.py`.
- Real cap-weighted index tracking: the repair HURTS a real S&P 500 weight snapshot
  (44.3% win rate, real return info the repair can't touch) --
  `experiments/index_tracking_repair_check.py`.
- Min-variance's harm is NOT (just) estimation noise: confirmed with a noise-free
  known-truth market that the harm survives a perfect covariance -- a real,
  provable second-order-optimality fact, NOT dependent on the streaming question --
  `experiments/known_truth_minvar_check.py`. (This one IS structurally solid.)
- Entrywise (not scalar) noise-aware correlation shrinkage as the repair TARGET:
  real but modest, same trade-off shape as the gamma-blend --
  `experiments/noise_aware_repair_check.py`.
- The classic "naive beats optimized under noise" flip is real on real data (needs
  T/n near 1) but the repair does NOT rescue it there -- repair quality itself
  degrades as T/n falls, in every universe-size block tested --
  `experiments/optim_vs_naive_noise_sweep.py`.
- Real-data tail-dependence at ES99 (not just ES95): still doesn't beat plain
  correlation repair, and during the actual COVID crash window the bootstrap tail
  overlay was clearly worse than correlation-only (19.4% win rate) --
  `experiments/real_tail_dependence_deep_check.py`, `experiments/covid_crash_tail_check.py`.
- Repaired HRP vs. simple alternatives (shrunk/Ledoit-Wolf min-variance, risk
  parity, a naive linear blend of weights): wins on raw one-shot performance, but
  the fair walk-forward re-check above complicates this --
  `experiments/repair_vs_simple_alternatives.py`.
- Options / other nonlinear-payoff portfolios as a cleaner, naturally-occurring
  path to genuine tail dependence (payoff convexity mechanically creates tail
  comovement from ordinary linear correlation, sidestepping the "need a crash
  example in-sample" problem) -- proposed, not yet built.

## Sources / reproducibility

- Method: builds directly on the ability-tilt machinery of the companion paper
  [thurstone-portfolios](../thurstone-portfolios/) (`cotton2026polish`) and the
  k-factor calibration engine in the sibling `winning` package
  (`cotton2026factorprobit`, `winning.factor.core`), wired into this repo via
  `allocation/_thurstone/factor.py`.
- Known-truth market: `experiments/bowling_sim.py` (already in this repo).
- Real markets: S&P 500 (2014-2024, 583 names) and REIT (2000-2024, 49 names) daily
  return panels, sourced from
  [winningportstudy](https://github.com/microprediction/winningportstudy) and cached
  as `experiments/data/sp500_returns_2014_2024.parquet` /
  `experiments/data/reit_returns.parquet` (not committed -- regenerate from that repo's
  `step_0/s500_ret0_part_*.csv` and `reitdata/reits_dly_ret.parquet`).
- Experiment code: `experiments/schur_thurstone_repair.py` (main correlation- and
  tail-repair sweep), `experiments/schur_gamma_sweep.py` (repair depth vs. Schur's
  gamma), `experiments/estimation_noise_sweep.py` (repair vs. in-sample window
  length), `experiments/overnight_repair_validation.py` (real-market robustness
  sweep, Section 4.5), `experiments/seriation_check.py` (Fiedler vs. classical
  single-linkage seriation, Section 4.5), `experiments/return_metrics_check.py`
  (mean/vol/Sharpe/ES95 decomposition, Section 4.5), `experiments/
  iterate_repair_check.py` (repeated-application stability, Discussion),
  `experiments/gamma_blend_check.py` (gamma-blended reference, Section 4.6),
  `experiments/tail_confidence_dial_check.py` (confidence-weighted tail view,
  Section 4.7). Each is runnable standalone from `experiments/`.

## Files

- `paper.tex` -- manuscript (imports `../shared/preamble.tex`, cites `../refs.bib`)
- `figures/` -- figures used in this paper (none yet)
- Build: `../build.sh repaired-hierarchical-portfolios` -> `paper.pdf`
