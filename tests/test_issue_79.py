"""Issue #79: the low-rank factor path must not depend on the asset order.

``factor_decompose`` used a positional Gaussian sketch, which a relabelling of
the assets regenerates rather than permutes, so ``C`` and ``P C P'`` gave
factor models that disagreed by 0.8 in an implied correlation after undoing
``P``. Every case here keeps ``k + oversample < n`` so the range finder, not a
full decomposition, is what is exercised.
"""

import numpy as np

from allocation._thurstone.covariance import factor_decompose
from allocation._thurstone.transport import transport_weights_lowrank


def _issue_corr(n=12, seed=45):
    A = np.random.default_rng(seed).normal(size=(n, 10))
    C = A @ A.T + 0.2 * np.eye(n)
    return C / np.sqrt(np.outer(np.diag(C), np.diag(C)))


def _implied(C, k):
    B, d = factor_decompose(C, k, seed=0)
    return B, d, B @ B.T + np.diag(d)


def test_factor_covariance_is_permutation_equivariant():
    for n, seed, k in [(12, 45, 2), (40, 1, 3), (150, 2, 4)]:
        C = _issue_corr(n, seed)
        perm = np.random.default_rng(1001).permutation(n)
        inv = np.argsort(perm)
        _, _, M = _implied(C, k)
        _, _, Mp = _implied(C[np.ix_(perm, perm)], k)
        assert np.allclose(np.diag(M), 1.0)
        assert np.max(np.abs(M - Mp[np.ix_(inv, inv)])) < 1e-8


def test_low_rank_race_weights_follow_the_relabelling():
    n, k, m = 12, 2, 200_000
    C = _issue_corr(n)
    perm = np.random.default_rng(1001).permutation(n)
    inv = np.argsort(perm)
    B, d, _ = _implied(C, k)
    Bp, dp, _ = _implied(C[np.ix_(perm, perm)], k)
    rng = np.random.default_rng(3)
    zf, zi = rng.standard_normal((m, k)), rng.standard_normal((m, n))
    a = np.zeros(n)
    w = transport_weights_lowrank(a, B, d, zf, zi)
    # idiosyncratic seeds travel with their assets; a factor's sign is free,
    # so the two races agree up to Monte Carlo noise rather than pathwise
    wp = transport_weights_lowrank(a, Bp, dp, zf, zi[:, perm])[inv]
    assert np.max(np.abs(w - wp)) < 0.01
