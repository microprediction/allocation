"""Cross-check the Schur coupling port against skfolio's reference implementation.

Skipped unless skfolio is installed. On well-conditioned blocks (where neither
side needs SPD repair) the two agree to machine precision, confirming the port of
the gamma augmentation + recursive bisection is faithful.
"""

import numpy as np
import pytest

pytest.importorskip("skfolio")

from skfolio.optimization.cluster.hierarchical._schur import (  # noqa: E402
    _compute_weights as sk_compute_weights,
)
from skfolio.optimization.cluster.hierarchical._schur import (  # noqa: E402
    _schur_augmentation as sk_augmentation,
)

from allocation._schur.coupling import compute_weights, schur_augmentation  # noqa: E402


def test_augmentation_matches_skfolio():
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(50):
        m = int(rng.integers(4, 12))
        na, nd = m // 2, m - m // 2
        if rng.random() < 0.5:
            na, nd = nd, na
        a = rng.standard_normal((na, na))
        a = a @ a.T + np.eye(na)
        d = rng.standard_normal((nd, nd))
        d = d @ d.T + np.eye(nd)
        b = rng.standard_normal((na, nd)) * 0.3
        g = float(rng.uniform(0, 1))
        max_err = max(max_err, np.max(np.abs(schur_augmentation(a, b, d, g) - sk_augmentation(a, b, d, g))))
    assert max_err < 1e-12


def test_compute_weights_matches_skfolio():
    rng = np.random.default_rng(1)
    max_err, compared = 0.0, 0
    for _ in range(60):
        n = int(rng.integers(4, 14))
        A = rng.standard_normal((n, 3 * n))
        cov = A @ A.T / (3 * n) + np.diag(rng.uniform(0.3, 0.8, n))  # well conditioned
        order = rng.permutation(n)
        for gamma in (0.0, 0.25, 0.5, 0.75):
            mine = compute_weights(order, cov, gamma, force_spd=True)
            try:
                theirs = sk_compute_weights(
                    gamma, order, cov.copy(),
                    max_weights=np.ones(n), min_weights=np.zeros(n), force_spd=True,
                )
            except Exception:
                continue  # skfolio raises on ill-conditioned blocks; skip those
            max_err = max(max_err, np.max(np.abs(mine - theirs)))
            compared += 1
    assert compared > 100
    assert max_err < 1e-10
