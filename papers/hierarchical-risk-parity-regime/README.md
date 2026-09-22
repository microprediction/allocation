# Hierarchical Risk Parity Has No Regime of Advantage

Tests the defence of hierarchical risk parity on its own terms, with the
population covariance known by construction so realized risk is exact, and
finds no interval of sample size where the method is the right choice.

Below a fifth of an observation per asset it is beaten by inverse-variance
weighting, which also inverts nothing. Above parity it is beaten by any
regularised optimizer, and by long-only minimum variance on the raw sample in
76 percent of five hundred random structures, so the no-short-sale constraint
alone closes the gap.

Supporting results: no shrinkage class with fewer than `n-1` parameters can
reproduce the rule, confirmed at three universe sizes; and nested clustered
optimization on the same clusters tracks the shrinkage frontier while
hierarchical risk parity does not, which puts the cost in the split rule
rather than in the hierarchy.

Reproduce with `experiments/studies` and `experiments/benchmark`; the numbers
are recorded in `experiments/studies/RESULTS.md` sections 1 to 6 and 9.

Build: `./build.sh hierarchical-risk-parity-regime`
