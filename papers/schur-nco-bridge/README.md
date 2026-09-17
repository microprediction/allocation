# A Schur Bridge from Nested Clustered Optimization to Global Minimum Variance

- **Status**: draft
- **Authors**: Peter Cotton
- **Started**: 2026-09-16

## Abstract

Nested clustered optimization (NCO) allocates within each cluster from the
cluster's own covariance block and then across the resulting cluster
portfolios. Block inversion says the unconstrained minimum-variance
portfolio has the same two-tier shape, with each block replaced by its Schur
complement against every other asset. Conditioning instead on one
knot from each other cluster truncates the conditioning set in the manner
of a Vecchia approximation, and the paper gives the rank-one "gateway model" of cross-cluster dependence under
which it is exact. Damping the complement by gamma then gives a bridge with
NCO at gamma=0 and the global optimum at gamma=1, with no linear solve
larger than a cluster or the number of clusters.

## Scope

Exactness is for the unconstrained, fully invested, long-short problem, and
the two-tier form needs nonzero cluster totals.

## Files

- `paper.tex` — manuscript
- `verify_schur_nco_bridge.py` — certificate for all algebraic propositions
  and the listed identities; also run by
  `tests/test_schur_nco_bridge_certificate.py`
- `../refs.bib` — shared bibliography
