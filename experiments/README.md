# Experiments

Two directories with different jobs.

**`benchmark/`** is the harness: one market generator, one evaluation protocol,
every allocator in the package driven through the interface a user would use.
Point it at the package and it tells you which method wins, by how much, with a
standard error, and how often each one failed to fit. Use this for any "how do
these compare" question.

**`studies/`** are the one-off results behind `studies/RESULTS.md`. Each answers
a specific question the harness cannot, usually by holding the market fixed and
sweeping one thing. They keep their own fixed markets in `studies/_markets.py`
so their recorded numbers stay reproducible.

## Running things

```
python run_all.py --quick      # every fast study, a few minutes
python run_all.py --full       # everything, roughly an hour
python run_all.py --list       # what exists and what it costs
cd benchmark && python run.py 60 regime
```

## Costs

The Thurstone race is cheap now. Every configuration is 0.02 to 0.07 seconds a
fit at forty to a hundred assets, including the market calibration, which used
to be the one setting to avoid. The path budget is nearly free, so use a large
one: the harness runs at 65,536 paths.

That was not true until the calibration was routed through `winning`'s front
door. It had been going through a deprecated alias to the density-agnostic
engine, which the compiled kernels do not touch, and a fit cost about 1.4
seconds with the market calibration at 12.9 seconds at forty assets and 100
seconds at a hundred. The tilt studies below were sized against those numbers
and are now far faster than their labels suggest.

| study | what it establishes | cost |
|---|---|---|
| `implied.py` | every portfolio is minimum variance on some positive definite matrix; the distortion of each allocator | seconds |
| `dimension.py` | no shrinkage class with fewer than n-1 parameters reproduces a portfolio; the cliff sits exactly there | ~2 min |
| `closest.py`, `what_shrinkage.py` | which named shrinkage comes closest to HRP, and that the answer is near-diagonal | ~3 min |
| `decompose.py` | HRP's gap to minimum variance split into information discarded and rule applied | ~1 min |
| `bridge_distortion.py`, `psd_gamma.py` | distortion falls monotonically along the coupling dial; where the implied covariance enters the PSD cone | ~2 min |
| `nested_best_case.py` | HRP on a clean nested hierarchy, its best case, where it loses by a factor of two | ~4 min |
| `taper_beats_hrp.py` | HRP sits at the dominated corner of a two-parameter family containing it | ~10 min |
| `nco_taper.py` | NCO tracks the shrinkage frontier and HRP does not | ~8 min |
| `confound.py`, `rank_deficient.py`, `robust.py`, `shrinkage_beats_hrp.py` | the comparisons now covered by `benchmark/run.py`; kept because `RESULTS.md` quotes their numbers | 2-15 min |
| `thurstone_paired.py` | the tilt is neutral on a stationary market | ~1 min |
| `tail_test.py` | the tilt on a regime market, scored on variance and shortfall | ~2 min |
| `tilt_confirm.py`, `tilt_replicate.py`, `tilt_highdata.py` | three seeds; the tilt is worth nothing at T/n=0.1 and 13% of variance at T/n=5 | ~1 min each |
| `mhp.py` | the multi-hypothesis ensemble method, implemented and benchmarked | ~5 min |
| `nonelliptical.py` | that a market can separate tail risk from variance, now also in `benchmark/markets.py` | ~1 min |

## Read this before trusting a new result

`studies/RESULTS.md` ends with three measurements that were thrown away. Each
produced a clean-looking table that meant nothing: a Monte Carlo budget smaller
than the effect, an elliptical market on which expected shortfall is
algebraically a function of variance, and a regime mixture that stayed
effectively elliptical. They are kept because each was a bad instrument rather
than a bad method, and that distinction is the first thing to get lost.

A positive result also deserves a fresh-seed replication. In this work the
first replication pulled all four shared cells down and killed one outright,
though every one of them passed a two-sigma check.
