"""#60: long_only=True must never return a short portfolio.

The frontier caps eta within each leaf only; negative conditioned companions
(gamma > 0) and NCO's unconstrained outer step can still short. The bridge now
either returns nonnegative weights or raises.
"""

import numpy as np
import pytest

from allocation import SchurBridge
from allocation._schur.bridge import bridge_weights

S2 = np.array([[1.0, 1.5], [1.5, 4.0]])  # positive definite; min variance shorts
SINGLETONS = [np.array([0]), np.array([1])]


def _shorting_cov(n=8, seed=0):
    rng = np.random.default_rng(seed)
    sig = rng.uniform(0.5, 5.0, n)
    R = 0.6 * np.ones((n, n)) + 0.4 * np.eye(n)
    return np.outer(sig, sig) * R


@pytest.mark.parametrize("conditioning", ["siblings", "all"])
def test_issue_example_nco_at_gamma_zero_raises(conditioning):
    with pytest.raises(ValueError, match="long_only"):
        bridge_weights(S2, SINGLETONS, gamma=0.0, long_only=True,
                       outer="optimize", conditioning=conditioning)


@pytest.mark.parametrize("conditioning", ["siblings", "all"])
def test_singleton_leaves_with_negative_companion_raise(conditioning):
    with pytest.raises(ValueError, match="long_only"):
        bridge_weights(S2, SINGLETONS, gamma=1.0, long_only=True,
                       outer="stack", conditioning=conditioning)


@pytest.mark.parametrize("conditioning", ["siblings", "all", "factor"])
@pytest.mark.parametrize("outer", ["stack", "optimize"])
@pytest.mark.parametrize("n_clusters", [8, 4, 2])
@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_long_only_weights_are_nonnegative_or_raise(conditioning, outer, n_clusters, gamma):
    cov = _shorting_cov()
    assert np.any(np.linalg.solve(cov, np.ones(8)) < 0)  # the signed optimum shorts
    part = [np.asarray(c) for c in np.array_split(np.arange(8), n_clusters)]
    try:
        w = bridge_weights(cov, part, gamma=gamma, eta=1.0, conditioning=conditioning,
                           outer=outer, long_only=True)
    except ValueError:
        return
    assert np.all(w >= -1e-12)
    assert np.isclose(w.sum(), 1.0)


@pytest.mark.parametrize("conditioning", ["siblings", "all", "factor"])
def test_stack_at_gamma_zero_is_long_only(conditioning):
    cov = _shorting_cov()
    part = [np.arange(4), np.arange(4, 8)]
    w = bridge_weights(cov, part, gamma=0.0, eta=1.0, conditioning=conditioning,
                       outer="stack", long_only=True)
    assert np.all(w >= -1e-12)


def test_estimator_raises_instead_of_returning_shorts():
    X = np.random.default_rng(1).multivariate_normal(np.zeros(2), S2, size=400)
    est = SchurBridge(gamma=0.0, clusters=np.array([0, 1]), conditioning="all",
                      outer="optimize", long_only=True)
    with pytest.raises(ValueError, match="long_only"):
        est.fit(X)
