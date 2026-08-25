# Repaired Hierarchical Portfolios

- **Status**: draft
- **Authors**: Peter Cotton
- **Started**: 2026-08-25

## Abstract

HRP and Schur-complementary allocation build a portfolio by recursive bisection that
only ever consults within-block covariance (and, at Schur's gamma=0, HRP itself,
never any cross-block covariance at all). We repair this without re-optimizing:
calibrate the latent Thurstone abilities that reproduce the hierarchical weights
under a reconstruction of the dependence the recursion actually used (a nested
factor model implied by the same median-bisection tree), then re-race those scores
under the fuller dependence on hand. On a known-truth contagious market, the
correlation-completion repair cuts out-of-sample ES95 by 9-17% versus plain HRP and
Schur, survives sample sizes down to two observations per asset, and the amount of
repair that helps shrinks as Schur's own gamma rises -- consistent with the repair
filling a genuine gap rather than double-counting information. Replicated on 12,000
random trials against real S&P 500 and REIT sub-portfolios (varying estimation
window, covariance estimator, sub-portfolio, and gamma): correlation completion wins
59-80% of trials across every estimator tried, and needs LESS restored correlation
on real (noisily estimated) data than on the synthetic market, not more -- the
race-reallocation effect, not correlation recovery per se, turns out to be the
dominant and most robust part of the gain. A sharper test of whether tail dependence
specifically adds further value did not confirm the hypothesis on the synthetic
market or in aggregate on real data, but split cleanly by universe on real data: a
modest win on the sector-concentrated REIT universe, a net loss on broad S&P 500
sub-portfolios. Both results are reported rather than the one that flatters the
method. Decomposing the real-market gain by statistic (1,500 further trials) shows
it is specifically a volatility/tail-risk reduction, not a return forecast: mean
return win rate is 46.9% (indistinguishable from a coin flip, exactly as the
construction predicts, since neither the base allocators nor the repair ever
consult a return estimate), while volatility (78-83%) and ES95 (75-80%) move
reliably. Finally, the repair should be applied once, not iterated: it has no
fixed point of its own (only C_used=C_full is a fixed point, which F=0 never is),
and feeding its own output back in repeatedly drives a weight to zero -- confirmed
at 32x Monte Carlo resolution to be genuine, not a sampling artifact -- typically
within 3-6 rounds, on the same real correlated pair each time it recurs.

## Status / TODO

Early draft. Numbers in the paper are real (not placeholders): the synthetic
(bowling-lab) tables are produced by the three scripts below at reduced
quadrature/Monte-Carlo resolution for exploratory speed; the real-market table is
from a 12,000-trial, ~6.4-hour overnight run with zero errors
(`experiments/overnight_repair_validation.py`, `experiments/data/overnight_results.csv`,
not committed -- see `.gitignore`).

- [ ] Re-run key synthetic tables at production resolution (full points, Q, Sobol budget).
- [ ] Try the alternative `C_used` constructions floated in the discussion section
      (flat K-cluster block-diagonal with exact within-cluster covariance; a
      gamma-blended reference `(1-gamma)*C_tree + gamma*Sigma_hat` for Schur --
      the real-market gamma-sweep result now gives a concrete reason to build this).
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
- [ ] Figures: ES95 vs. max_factors curves; gain vs. T_in; win rate by estimator.

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
  iterate_repair_check.py` (repeated-application stability, Discussion). Each is
  runnable standalone from `experiments/`.

## Files

- `paper.tex` -- manuscript (imports `../shared/preamble.tex`, cites `../refs.bib`)
- `figures/` -- figures used in this paper (none yet)
- Build: `../build.sh repaired-hierarchical-portfolios` -> `paper.pdf`
