import numpy as np
import pytest

from allocation import (
    MinimumVariance,
    SchurComplementary,
    StreamingSchur,
    StreamingTurnoverPenalty,
    TurnoverPenalty,
)


def _returns(n_obs=600, n=8, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n_obs, 2))
    load = np.zeros((n, 2))
    load[: n // 2, 0] = rng.uniform(0.4, 0.9, n // 2)
    load[n // 2 :, 1] = rng.uniform(0.4, 0.9, n - n // 2)
    return f @ load.T + 0.5 * rng.standard_normal((n_obs, n))


def test_alpha_and_budget():
    est = TurnoverPenalty(MinimumVariance(), cost=3.0)
    assert est.alpha_ == pytest.approx(0.25)
    w = est.fit(_returns()).weights_
    assert abs(float(w.sum()) - 1.0) < 1e-9


def test_cost_zero_is_identity():
    X0, X1 = _returns(), _returns(n_obs=5, seed=1)
    base = MinimumVariance().fit(X0)
    base.partial_fit(X1)
    wrapped = TurnoverPenalty(MinimumVariance(), cost=0.0).fit(X0)
    wrapped.partial_fit(X1)
    assert np.allclose(base.weights_, wrapped.weights_)


def test_turnover_scaled_by_alpha():
    # on the first partial_fit, prev weights match the base, so the damped step
    # is exactly alpha times the undamped step.
    X0, X1 = _returns(), _returns(n_obs=5, seed=1)
    cost = 4.0
    alpha = 1.0 / (1.0 + cost)

    base = MinimumVariance().fit(X0)
    w0 = base.weights_.copy()
    base.partial_fit(X1)
    step_base = np.max(np.abs(base.weights_ - w0))

    wrapped = TurnoverPenalty(MinimumVariance(), cost=cost).fit(X0)
    assert np.allclose(wrapped.weights_, w0)  # cold start = base
    wrapped.partial_fit(X1)
    step_wrapped = np.max(np.abs(wrapped.weights_ - w0))

    assert step_wrapped == pytest.approx(alpha * step_base, rel=1e-9)


def test_higher_cost_means_less_turnover():
    X = [_returns()] + [_returns(n_obs=5, seed=s) for s in range(1, 6)]
    steps = {}
    for cost in (0.0, 1.0, 9.0):
        est = TurnoverPenalty(SchurComplementary(gamma=0.5), cost=cost).fit(X[0])
        prev = est.weights_.copy()
        total = 0.0
        for Xi in X[1:]:
            est.partial_fit(Xi)
            total += float(np.sum(np.abs(est.weights_ - prev)))
            prev = est.weights_.copy()
        steps[cost] = total
    assert steps[0.0] > steps[1.0] > steps[9.0]


def test_predict_and_get_params():
    est = TurnoverPenalty(MinimumVariance(), cost=2.0).fit(_returns())
    r = est.predict(_returns(n_obs=10, seed=2))
    assert r.shape == (10,)
    assert set(est.get_params()) == {"estimator", "cost"}


def test_streaming_turnover_penalty_reduces_churn_changing_universe():
    def run(cost):
        rng = np.random.default_rng(2)
        est = StreamingTurnoverPenalty(StreamingSchur(gamma=0.5, min_obs=10), cost=cost)
        prev, total = {}, 0.0
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
                assert abs(sum(w.values()) - 1.0) < 1e-6
            if t > 60 and set(w) == set(prev):
                total += sum(abs(w[k] - prev[k]) for k in w)
            prev = w
        return total

    assert run(5.0) < run(0.0)
