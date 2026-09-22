# Benchmark harness

One market generator, one evaluation protocol, every allocator in the package
driven through the same interface. Replaces a pile of ad-hoc scripts that each
reimplemented HRP and minimum variance slightly differently and each invented
their own market.

```
cd experiments/benchmark
python run.py [draws] [market] [reference]
```

`market` is one of `gaussian`, `ar`, `student-t`, `regime`. `reference` is any
method name; everything is reported as a ratio to it with a paired win rate.

## What it fixes

**It benchmarks what ships.** Every method is the package class, constructed and
called as a user would. Three defects showed up within minutes of pointing it at
the package: issue #47, where `SchurComplementary` fails inside `fit` and the
error blames the caller, and issue #48, where `SchurBridge` silently returns
equal weight on a singular covariance, which in a table reads as a result
rather than as a fallback.

**It reports uncertainty.** Every win rate carries a standard error. A 64% rate
on 25 draws is 1.5 standard errors from a coin.

**It pairs.** Every method sees the identical sample on every draw, and win
rates are computed per draw rather than from marginal medians, which can
disagree with the pairing.

**It says who is long-only.** Comparing a long-only heuristic against an
unconstrained optimizer measures the constraint as much as the rule. The column
is there so that is never accidental.

**It makes failure visible.** A method that cannot fit is recorded as missing
and its coverage is printed beside its score, rather than being replaced by a
fallback that flatters it.

**It checks the market can answer the question.** `markets.elliptical_spread`
reports whether a tail metric adds anything over variance on that market.
Multivariate t scores about 1.01, meaning expected shortfall is a relabelling
of variance there and cannot test tail skill. The regime market scores about
1.15. `run.py` prints this and only reports the shortfall table when it is
above 1.05.

## Three traps these modules exist to avoid

Each was hit at least once, and each produced a clean-looking table that meant
nothing.

1. **A Monte Carlo budget smaller than the effect.** A simulated allocator's
   weights carry noise scaling as `1/sqrt(paths)`. Pair under common random
   numbers and print a fidelity check at the setting that should reproduce a
   known target exactly.
2. **An elliptical market cannot test tail skill.** Shortfall is proportional
   to standard deviation, so the tail column is the variance column relabelled.
3. **A regime mixture that pushes all correlations uniformly toward one stays
   effectively elliptical.** The crash regime needs its own independent
   structure over a different labelling.

## Files

| file | what it holds |
|---|---|
| `markets.py` | six covariance families, gaussian / AR(1) / student-t / regime samplers, and the elliptical check |
| `methods.py` | the registry, with long-only and source recorded per method |
| `protocol.py` | paired evaluation and the summary with standard errors and coverage |
| `run.py` | the driver |
