import numpy as np
import pytest

from allocation import (
    BoxConstrained,
    InverseVariance,
    MinimumVariance,
    SchurComplementary,
    StreamingBoxConstrained,
    StreamingSchur,
)
from allocation.constraints import barrier_constrain


def _returns(n_obs=600, n=8, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n_obs, 2))
    load = np.zeros((n, 2))
    load[: n // 2, 0] = rng.uniform(0.4, 0.9, n // 2)
    load[n // 2 :, 1] = rng.uniform(0.4, 0.9, n - n // 2)
    return f @ load.T + 0.5 * rng.standard_normal((n_obs, n))


# ---------------------------------------------------------------- solver

def test_box_respected_and_simplex():
    rng = np.random.default_rng(0)
    target = rng.dirichlet(np.ones(8))
    w = barrier_constrain(target, lower=0.0, upper=0.2)
    assert abs(w.sum() - 1.0) < 1e-6
    assert np.all(w >= -1e-9) and np.all(w <= 0.2 + 1e-9)


def test_feasible_target_barely_moves():
    # a target already comfortably inside the box should be returned ~unchanged
    target = np.full(8, 1.0 / 8)  # well within [0, 0.5]
    w = barrier_constrain(target, lower=0.0, upper=0.5, tau=1e-5)
    assert np.max(np.abs(w - target)) < 5e-3


def test_lower_bound_floor():
    target = np.array([0.9, 0.04, 0.03, 0.03])
    w = barrier_constrain(target, lower=0.1, upper=1.0)
    assert np.all(w >= 0.1 - 1e-9)
    assert abs(w.sum() - 1.0) < 1e-6


def test_group_caps_disjoint():
    rng = np.random.default_rng(1)
    target = rng.dirichlet(np.ones(6))
    groups = [[0, 1, 2], [3, 4, 5]]
    caps = [0.4, 0.8]
    w = barrier_constrain(target, lower=0.0, upper=1.0, groups=groups, group_caps=caps)
    assert abs(w.sum() - 1.0) < 1e-6
    assert w[[0, 1, 2]].sum() <= 0.4 + 1e-6
    assert w[[3, 4, 5]].sum() <= 0.8 + 1e-6
    assert np.all(w >= -1e-9)


def test_infeasible_box_raises():
    with pytest.raises(ValueError):
        barrier_constrain(np.full(4, 0.25), lower=0.0, upper=0.1)  # 4*0.1 < 1


# -------------------------------------------------------------- batch wrapper

@pytest.mark.parametrize("Est", [MinimumVariance, InverseVariance, SchurComplementary])
def test_wrapper_caps_each_name(Est):
    X = _returns()
    est = BoxConstrained(Est(), lower=0.0, upper=0.2).fit(X)
    w = est.weights_
    assert abs(w.sum() - 1.0) < 1e-6
    assert np.all(w <= 0.2 + 1e-9) and np.all(w >= -1e-9)


def test_wrapper_smooth_and_predict():
    X = _returns()
    est = BoxConstrained(SchurComplementary(gamma=0.5), upper=0.25).fit(X)
    before = est.weights_.copy()
    est.partial_fit(_returns(n_obs=5, seed=1))
    assert np.max(np.abs(est.weights_ - before)) < 0.1  # low turnover
    assert est.predict(_returns(n_obs=10, seed=2)).shape == (10,)
    assert set(est.get_params()) == {"estimator", "lower", "upper", "groups", "group_caps", "tau"}


# ------------------------------------------------------------ streaming wrapper

def test_streaming_box_caps_changing_universe():
    rng = np.random.default_rng(2)
    est = StreamingBoxConstrained(StreamingSchur(gamma=0.5, min_obs=10), upper=0.35)
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
            assert all(v <= 0.35 + 1e-6 for v in w.values())
    assert "D" in est.weights and "A" not in est.weights


def test_streaming_group_caps_by_label():
    rng = np.random.default_rng(3)
    labels = {"A": "x", "B": "x", "C": "y", "D": "y"}
    est = StreamingBoxConstrained(
        StreamingSchur(gamma=0.5, min_obs=10),
        upper=1.0, groups=labels, group_caps={"x": 0.5},
    )
    assets = ["A", "B", "C", "D"]
    for _ in range(120):
        f = rng.standard_normal(2)
        x = {a: 0.6 * f[i % 2] + 0.5 * rng.standard_normal() for i, a in enumerate(assets)}
        est.learn_one(x)
    w = est.predict_one()
    assert w["A"] + w["B"] <= 0.5 + 1e-6
    assert abs(sum(w.values()) - 1.0) < 1e-6
