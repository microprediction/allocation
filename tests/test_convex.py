import numpy as np
import pytest

from allocation import (
    EqualWeight,
    MaximumDecorrelation,
    MaximumDiversification,
    MeanVariance,
    MinimumVariance,
    StreamingMaximumDecorrelation,
    StreamingMaximumDiversification,
    StreamingMeanVariance,
    StreamingMinimumVariance,
)
from allocation.convex import (
    max_decorrelation_weights,
    max_diversification_weights,
    mean_variance_weights,
    min_variance_weights,
)


def _returns(n_obs=600, n=8, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n_obs, 2))
    load = np.zeros((n, 2))
    load[: n // 2, 0] = rng.uniform(0.4, 0.9, n // 2)
    load[n // 2 :, 1] = rng.uniform(0.4, 0.9, n - n // 2)
    return f @ load.T + 0.5 * rng.standard_normal((n_obs, n))


def _cov(X):
    return np.cov(X, rowvar=False)


def _sums_to_one(w):
    return abs(float(np.sum(w)) - 1.0) < 1e-9


# ------------------------------------------------------------ closed forms

def test_min_variance_on_diagonal_is_inverse_variance():
    var = np.array([0.04, 0.01, 0.09, 0.25])
    cov = np.diag(var)
    w = min_variance_weights(cov)
    inv = 1.0 / var
    assert np.allclose(w, inv / inv.sum())


def test_max_diversification_on_diagonal_is_inverse_vol():
    var = np.array([0.04, 0.01, 0.09, 0.25])
    cov = np.diag(var)
    w = max_diversification_weights(cov)
    inv_vol = 1.0 / np.sqrt(var)
    assert np.allclose(w, inv_vol / inv_vol.sum())


def test_min_variance_beats_equal_weight_variance():
    cov = _cov(_returns())
    w_mv = min_variance_weights(cov)
    n = len(cov)
    w_eq = np.full(n, 1.0 / n)
    assert w_mv @ cov @ w_mv < w_eq @ cov @ w_eq
    assert _sums_to_one(w_mv)


def test_shrinkage_one_is_equal_weight():
    cov = _cov(_returns())
    n = len(cov)
    assert np.allclose(min_variance_weights(cov, shrinkage=1.0), np.full(n, 1.0 / n))


def test_max_decorrelation_on_diagonal_is_equal_weight():
    # diagonal covariance -> correlation is the identity -> equal weight
    cov = np.diag([0.04, 0.01, 0.09, 0.25])
    assert np.allclose(max_decorrelation_weights(cov), np.full(4, 0.25))


def test_mean_variance_on_diagonal():
    var = np.array([0.04, 0.01, 0.09, 0.25])
    mu = np.array([0.1, 0.2, 0.05, 0.15])
    w = mean_variance_weights(np.diag(var), mu)
    raw = mu / var  # Sigma^{-1} mu with diagonal Sigma
    assert np.allclose(w, raw / raw.sum())


def test_mean_variance_uses_ewma_mean_when_unsupplied():
    X = _returns()
    est = MeanVariance().fit(X)  # mu defaults to the EWMA running mean
    assert _sums_to_one(est.weights_) and np.all(np.isfinite(est.weights_))
    # supplying mu explicitly is honoured
    mu = np.ones(X.shape[1])
    w = MeanVariance(expected_returns=mu).fit(X).weights_
    assert _sums_to_one(w)


@pytest.mark.parametrize("Est", [MinimumVariance, MaximumDiversification, MaximumDecorrelation])
def test_estimator_sums_to_one_and_smooth(Est):
    X = _returns()
    est = Est(shrinkage=0.1).fit(X)
    assert _sums_to_one(est.weights_)
    before = est.weights_.copy()
    est.partial_fit(_returns(n_obs=5, seed=1))
    assert np.max(np.abs(est.weights_ - before)) < 0.1  # low turnover


def test_mean_variance_is_smooth_with_fixed_mu():
    # with mu held fixed, the map is smooth in Sigma alone; the documented
    # caveat is that a *noisy* mu (the default EWMA mean) breaks this.
    X = _returns()
    mu = np.linspace(0.05, 0.2, X.shape[1])
    est = MeanVariance(expected_returns=mu, shrinkage=0.1).fit(X)
    before = est.weights_.copy()
    est.partial_fit(_returns(n_obs=5, seed=1))
    assert np.max(np.abs(est.weights_ - before)) < 0.1


def test_min_variance_handles_singular_cov():
    # rank-deficient covariance (more assets than obs) must not raise
    X = _returns(n_obs=4, n=8, seed=3)
    w = MinimumVariance().fit(X).weights_
    assert _sums_to_one(w) and np.all(np.isfinite(w))


# ------------------------------------------------------------- streaming

@pytest.mark.parametrize(
    "make",
    [
        StreamingMinimumVariance,
        StreamingMaximumDiversification,
        StreamingMaximumDecorrelation,
        StreamingMeanVariance,
    ],
)
def test_streaming_changing_universe(make):
    rng = np.random.default_rng(2)
    est = make(min_obs=10)
    for t in range(160):
        active = ["A", "B", "C", "E"]
        if t >= 40:
            active = active + ["D"]
        if t >= 110:
            active = [a for a in active if a != "A"]
        f = rng.standard_normal(2)
        x = {a: 0.6 * f[i % 2] + 0.5 * rng.standard_normal() for i, a in enumerate(active)}
        est.learn_one(x)
        w = est.predict_one()
        assert set(w).issubset(set(active))
        if w:
            assert abs(sum(w.values()) - 1.0) < 1e-6  # signed: sums to one, may be negative
    assert "D" in est.weights and "A" not in est.weights
