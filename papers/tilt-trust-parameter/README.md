# A Tilt Is Worth What the Correlation Is Worth

A long-only portfolio and a vector of latent abilities are the same object in
two charts. Re-running the race under an estimated correlation, damped by a
parameter, tilts any benchmark without inverting anything.

Measures what that dial is worth. It costs nineteen percent of variance at a
tenth of an observation per asset and saves thirteen at five, with the optimum
walking monotonically between, which is the behaviour of a trust parameter.

Two results point away from the obvious extensions. The effect is flat on a
stationary market and appears only where the covariance switches regime, so it
acts on structure a single covariance cannot hold. And the tail-aware
Student-t race is consistently worse than the Gaussian one, on both variance
and shortfall, on a market built to have asymmetric tail dependence.

Reproduce with `experiments/studies/thurstone_paired.py`, `tail_test.py` and
the three `tilt_*.py` seeds; recorded in `experiments/studies/RESULTS.md`
sections 7, 8 and 8b.

Build: `./build.sh tilt-trust-parameter`
