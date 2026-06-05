import numpy as np
import pytest

from allocation import (
    EqualWeight,
    HierarchicalRiskParity,
    InverseVariance,
    RiskParity,
    SchurComplementary,
    StreamingEqualWeight,
    StreamingHRP,
    StreamingInverseVariance,
    StreamingRiskParity,
)
from allocation.baselines import risk_parity_weights


def _returns(n_obs=600, n=8, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n_obs, 2))
    load = np.zeros((n, 2))
    load[: n // 2, 0] = rng.uniform(0.4, 0.9, n // 2)
    load[n // 2 :, 1] = rng.uniform(0.4, 0.9, n - n // 2)
    return f @ load.T + 0.5 * rng.standard_normal((n_obs, n))


def _cov(X):
    return np.cov(X, rowvar=False)


def _is_simplex(w):
    return np.all(np.asarray(w) >= -1e-9) and abs(float(np.sum(w)) - 1.0) < 1e-6


# ----------------------------------------------------------------- batch

@pytest.mark.parametrize("Est", [EqualWeight, InverseVariance, RiskParity, HierarchicalRiskParity])
def test_batch_long_only_simplex(Est):
    X = _returns()
    w = Est().fit(X).weights_
    assert len(w) == X.shape[1]
    assert _is_simplex(w)


def test_equal_weight_is_uniform():
    X = _returns(n=8)
    w = EqualWeight().fit(X).weights_
    assert np.allclose(w, 1.0 / 8)


def test_inverse_variance_matches_diagonal():
    # weights are 1/variance off the estimator's own (EWMA) covariance
    X = _returns()
    est = InverseVariance().fit(X)
    cov = est._cov_estimator.covariance_
    inv = 1.0 / np.diag(cov)
    expected = inv / inv.sum()
    assert np.allclose(est.weights_, expected)


def test_risk_parity_equalises_risk_contributions():
    cov = _cov(_returns())
    w, _ = risk_parity_weights(cov)
    rc = w * (cov @ w)  # risk contributions
    assert _is_simplex(w)
    assert np.all(w > 0)  # interior
    assert np.std(rc) / np.mean(rc) < 1e-4  # equal risk contributions


def test_hrp_equals_schur_gamma_zero():
    X = _returns()
    w_hrp = HierarchicalRiskParity().fit(X).weights_
    w_schur = SchurComplementary(gamma=0.0, keep_monotonic=False).fit(X).weights_
    assert np.allclose(w_hrp, w_schur)


@pytest.mark.parametrize("Est", [InverseVariance, RiskParity, HierarchicalRiskParity])
def test_partial_fit_is_smooth(Est):
    X = _returns()
    est = Est().fit(X)
    before = est.weights_.copy()
    est.partial_fit(_returns(n_obs=5, seed=1))
    assert np.max(np.abs(est.weights_ - before)) < 0.1  # low turnover


def test_risk_parity_warm_start_converges_to_cold():
    # warm-started and cold solves of the same covariance must agree
    cov = _cov(_returns())
    w_cold, _ = risk_parity_weights(cov)
    x0 = np.random.default_rng(0).uniform(0.1, 2.0, cov.shape[0])
    w_warm, _ = risk_parity_weights(cov, x0=x0)
    assert np.allclose(w_cold, w_warm, atol=1e-7)


# ------------------------------------------------------------- streaming

def _stream_changing_universe(est, n_obs=160, seed=2):
    rng = np.random.default_rng(seed)
    for t in range(n_obs):
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
            assert abs(sum(w.values()) - 1.0) < 1e-6
            assert all(v >= -1e-9 for v in w.values())
    return est


@pytest.mark.parametrize(
    "make",
    [
        lambda: StreamingEqualWeight(min_obs=10),
        lambda: StreamingInverseVariance(min_obs=10),
        lambda: StreamingRiskParity(min_obs=10),
        lambda: StreamingHRP(min_obs=10),
    ],
)
def test_streaming_changing_universe(make):
    est = _stream_changing_universe(make())
    assert "D" in est.weights and "A" not in est.weights


def test_streaming_risk_parity_is_smooth():
    rng = np.random.default_rng(1)
    est = StreamingRiskParity(min_obs=10)
    assets = ["A", "B", "C", "D"]
    prev, max_step = None, 0.0
    for t in range(200):
        f = rng.standard_normal()
        x = {a: 0.5 * f + 0.5 * rng.standard_normal() for a in assets}
        est.learn_one(x)
        w = est.predict_one()
        if t > 40 and prev and set(w) == set(prev):
            max_step = max(max_step, sum(abs(w[k] - prev[k]) for k in w))
        prev = w
    assert max_step < 0.1


def test_streaming_hrp_matches_streaming_schur_gamma_zero():
    from allocation import StreamingSchur

    a = _stream_changing_universe(StreamingHRP(min_obs=10))
    b = _stream_changing_universe(StreamingSchur(gamma=0.0, keep_monotonic=False, min_obs=10))
    assert set(a.weights) == set(b.weights)
    for k in a.weights:
        assert abs(a.weights[k] - b.weights[k]) < 1e-9
