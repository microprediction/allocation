# Conditioned Cores: A Schur Bridge from Nested Clustered Optimization to Global Minimum Variance

- **Status**: draft (rewritten 2026-09-17 after review; PRs #33 and #34 closed)
- **Authors**: Peter Cotton
- **Started**: 2026-09-16

## Abstract

Nested clustered optimization (NCO) allocates within each cluster from the
cluster's own covariance block and then across the resulting cluster
portfolios. Block inversion says the unconstrained minimum-variance
portfolio has the same two-tier shape, with each block replaced by its Schur
complement against every other asset. Under a rank-one "orbital model" of
cross-cluster dependence, the complement against everything equals the
complement against the other clusters' cores. Damping by gamma gives a
bridge with NCO at gamma=0 and the global optimum at gamma=1, with no
inverse larger than a cluster or the number of clusters. The cores are
borrowed from Bajo Traver's HCOA; the same model says when HCOA's strategic
use of the core covariance is accurate.

## Scope

Exactness is for the unconstrained, fully invested, long-short problem, and
the two-tier form needs nonzero cluster totals. HCOA is not an endpoint of
the bridge; it is another instance of the general two-stage form.

## Files

- `paper.tex` — manuscript
- `verify_conditioned_cores.py` — certificate for every proposition and
  remark; also run by `tests/test_conditioned_cores_certificate.py`
- `hcoa_as_recovered.md` / `.tex` — what is publicly recoverable about HCOA
  (Mario Bajo Traver, Banco de España, forthcoming JFDS, SSRN 5928122), with
  sources. Build the `.tex` with tectonic directly.
- `../refs.bib` — shared bibliography
