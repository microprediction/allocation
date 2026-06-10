import numpy as np
import pytest

from allocation import (
    FactorMaximumDiversification,
    FactorMinimumVariance,
    MinimumVariance,
)
from allocation.factor import (
    covariance_to_factors,
    factor_min_variance_weights,
    woodbury_solve,
)


def _factor_returns(n_obs=600, n=12, k=3, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n_obs, k))
    load = rng.uniform(-0.7, 0.9, size=(n, k))
    return f @ load.T + 0.5 * rng.standard_normal((n_obs, n))


def _sums_to_one(w):
    return abs(float(np.sum(w)) - 1.0) < 1e-9


def test_woodbury_matches_dense_inverse():
    rng = np.random.default_rng(0)
    n, k = 30, 4
    B = rng.standard_normal((n, k)) * 0.4
    psi = rng.uniform(0.2, 0.6, n)
    Sigma = B @ B.T + np.diag(psi)
    rhs = rng.standard_normal(n)
    assert np.allclose(woodbury_solve(B, psi, rhs), np.linalg.solve(Sigma, rhs), atol=1e-8)


def test_covariance_to_factors_is_positive_definite():
    cov = np.cov(_factor_returns(n=20, k=3), rowvar=False)
    B, psi = covariance_to_factors(cov, k=3)
    assert B.shape == (20, 3) and np.all(psi > 0)
    Sigma = B @ B.T + np.diag(psi)
    assert np.linalg.eigvalsh(Sigma)[0] > 0  # strictly PD -> invertible


def test_factor_min_variance_well_posed_when_rank_deficient():
    # 60 obs, 200 names: dense Sigma is rank-deficient and min-variance is undefined,
    # but the factor + Woodbury route is well-posed.
    X = np.random.default_rng(1).standard_normal((60, 200)) * 0.01
    est = FactorMinimumVariance(factors=5).fit(X)
    w = est.weights_
    cov = np.cov(X, rowvar=False)
    assert _sums_to_one(w) and np.all(np.isfinite(w))
    assert float(w @ cov @ w) > 0  # a real (positive) portfolio variance


def test_factor_min_variance_beats_equal_weight_variance():
    cov = np.cov(_factor_returns(n_obs=800, n=40, k=4), rowvar=False)
    B, psi = covariance_to_factors(cov, k=4)
    w = factor_min_variance_weights(B, psi)
    n = len(cov)
    assert w @ cov @ w < np.full(n, 1.0 / n) @ cov @ np.full(n, 1.0 / n)


def test_factor_modes_run_and_are_smooth():
    X = _factor_returns(n_obs=400, n=50, k=4)
    for Est in (FactorMinimumVariance, FactorMaximumDiversification):
        est = Est(factors=4).fit(X)
        assert _sums_to_one(est.weights_) and np.all(np.isfinite(est.weights_))
        before = est.weights_.copy()
        est.partial_fit(_factor_returns(n_obs=5, n=50, k=4, seed=9))
        assert np.max(np.abs(est.weights_ - before)) < 0.1


def test_uses_supplied_factor_loadings():
    # a covariance estimator exposing loadings_/idiosyncratic_ takes the O(n k) path
    class _StubFactorCov:
        def __init__(self):
            rng = np.random.default_rng(3)
            self.loadings_ = rng.standard_normal((8, 2)) * 0.3
            self.idiosyncratic_ = rng.uniform(0.2, 0.5, 8)
            self.covariance_ = self.loadings_ @ self.loadings_.T + np.diag(self.idiosyncratic_)

        def partial_fit(self, X, y=None):
            return self

        def fit(self, X, y=None):
            return self

    est = FactorMinimumVariance(covariance_estimator=_StubFactorCov()).fit(
        np.random.default_rng(0).standard_normal((50, 8))
    )
    stub = _StubFactorCov()
    expected = factor_min_variance_weights(stub.loadings_, stub.idiosyncratic_)
    assert np.allclose(est.weights_, expected, atol=1e-8)
