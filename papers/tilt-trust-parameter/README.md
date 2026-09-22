# A Tilt Is Worth What the Correlation Is Worth

A long-only portfolio and a vector of latent abilities are the same object in
two charts. Re-running the race under an estimated correlation, damped by a
parameter, tilts any benchmark without inverting anything.

Measures what that dial is worth. It costs nineteen percent of variance at a
tenth of an observation per asset and saves thirteen at five, with the optimum
walking monotonically between, which is the behaviour of a trust parameter.
The race is evaluated by quadrature rather than simulated, so the identity at
zero damping is exact to 1e-10.

Two things govern what the dial is worth, and they are separate. The first is
how much correlation the benchmark already accounts for. The race halves the
variance of equal weight and of inverse variance and moves a long-only
minimum-variance portfolio by a tenth of a percent, so it is a diversification
operator that is close to a fixed point on what is already diversified. The
second is whether the estimate deserves trust, which is what the dial
expresses.

One result points away from the obvious extension. The tail-aware Student-t
race is consistently worse than the Gaussian one, on both variance and
shortfall, on a market built to have asymmetric tail dependence.

Reproduce with `experiments/studies/thurstone_paired.py`, `tail_test.py` and
the three `tilt_*.py` seeds; recorded in `experiments/studies/RESULTS.md`
sections 7, 8 and 8b.

Build: `./build.sh tilt-trust-parameter`
