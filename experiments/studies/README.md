# HRP, shrinkage, and the Thurstone tilt

The simulation studies behind `RESULTS.md`. Every figure in that file is
`w' Sigma w` against the *true* covariance, so there is no backtest noise and
no oracle: the structures are simulated, the estimates come from finite
samples of them, and the scoring uses the generating process.

Run scripts from **inside this directory**, since they import each other:

```
cd experiments/hrp-shrinkage
python robust.py
```

## Run order

`confound.py` and `robust.py` are the foundations and everything else imports
from them. `confound.py` has the HRP implementation, the long-only and plain
minimum-variance solves and the block market. `robust.py` has the randomized
structure generator (six families), the cvxpy long-only solve, the clustering,
NCO and the cross-cluster taper. `tail_test.py` additionally exports the
crash-regime market used by the Thurstone replications.

| script | what it answers | section |
|---|---|---|
| `confound.py` | HRP vs min-var on raw and filtered covariance | 6 |
| `implied.py` | every portfolio is min-var on some PD matrix; distortion | 1, 2 |
| `dimension.py` | the cliff at k = n-1; no small class reproduces HRP | 1 |
| `closest.py` | which named shrinkage comes closest to HRP | 2 |
| `what_shrinkage.py` | fits of four shrinkage families to HRP's weights | 2 |
| `decompose.py` | HRP's penalty split into information vs rule | 3 |
| `taper_beats_hrp.py` | HRP sits at the dominated corner of a family containing it | 3 |
| `nco_taper.py` | NCO is on the shrinkage frontier, HRP is not | 4 |
| `shrinkage_beats_hrp.py` | Ledoit-Wolf long-only min-var against HRP | 5 |
| `robust.py` | 500 randomized structures, the headline robustness study | 5 |
| `rank_deficient.py` | T/n from 0.1 to 1.0, HRP's actual claim | 6 |
| `bridge_distortion.py`, `psd_gamma.py` | distortion along the Schur bridge | 1 |
| `nonelliptical.py` | validates that a market can separate tail from variance | traps |
| `thurstone_paired.py` | the tilt on a stationary market (neutral) | 7 |
| `thurstone_tail.py` | the tilt scored on shortfall, elliptical market (void) | traps |
| `tail_test.py` | the tilt on the crash market, both objectives | 8 |
| `tilt_confirm.py`, `tilt_replicate.py`, `tilt_highdata.py` | three seeds | 8b |

`thurstone_rank.py` holds `SampleCovariance`, the shim that gives Thurstone the
same `np.cov` every other method receives. Without it the comparison measures
the estimator rather than the allocator, which is the mistake the whole study
is about.

## Three measurements that failed

Logged at the end of `RESULTS.md` and worth reading before trusting any
simulation of this kind. A Monte Carlo budget smaller than the effect being
measured; an elliptical market, on which expected shortfall is algebraically a
function of variance and cannot test tail skill; and a regime mixture that
pushes all correlations uniformly toward one, which stays effectively
elliptical. Each produced a clean-looking table that meant nothing.

## Cost

`robust.py` is a few minutes. The Thurstone scripts are 15 to 40 minutes each,
dominated by the race at 16384 paths. `calib="market"` costs about 12 seconds a
fit at these sizes and is not used in any sweep.

## Sub-portfolio restriction

Four studies here ask how to form a sub-portfolio from a parent, and they are
the exploratory record rather than the answer.

| script | what it answers |
|---|---|
| `restriction.py` | can a race predict what an allocator does on a subset? |
| `restriction_block.py` | the same with the seriation held fixed and a whole block dropped |
| `market_restriction.py` | leveraging a market-found optimum onto sub-universes |
| `transport_vs_estimate.py` | transport the parent optimum, or estimate the sub-covariance? |

The scaled version lives in `research/subportfolio/`, which is where the
five-thousand-name runs and the committed results are. It is self-contained
and does not import from here.
