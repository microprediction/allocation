"""#51: barrier_constrain must reject infeasible constraints, never return an
infeasible portfolio, and handle feasible sets without a strict interior."""
import numpy as np
import pytest

from allocation import BoxConstrained, EqualWeight, StreamingBoxConstrained
from allocation.constraints import barrier_constrain
from allocation.keyed import StreamingEqualWeight


def test_full_universe_cap_below_one_raises():
    with pytest.raises(ValueError, match="infeasible"):
        barrier_constrain([0.5, 0.5], groups=[[0, 1]], group_caps=[0.9])


def test_infeasible_group_totals_raise():
    # each group capped at 0.4 cannot hold a total of one
    with pytest.raises(ValueError, match="infeasible"):
        barrier_constrain(np.full(4, 0.25), groups=[[0, 1], [2, 3]], group_caps=[0.4, 0.4])


def test_upper_bounds_summing_exactly_to_one_return_the_unique_point():
    w = barrier_constrain([0.8, 0.2], upper=0.5)
    assert np.allclose(w, [0.5, 0.5])


def test_fixed_coordinate_is_eliminated():
    w = barrier_constrain([0.2, 0.3, 0.5], lower=[0, 0, 0.5], upper=[1, 1, 0.5])
    assert np.all(np.isfinite(w))
    assert w[2] == 0.5
    assert abs(w.sum() - 1.0) < 1e-12
    assert np.all(w[:2] > 0)


def test_caps_forcing_a_face_without_interior_raise():
    with pytest.raises(ValueError, match="no strict interior"):
        barrier_constrain([0.7, 0.1, 0.1, 0.1], groups=[[0, 1], [2, 3]], group_caps=[0.5, 0.5])


def test_tight_but_strictly_feasible_disjoint_groups():
    caps = [0.501, 0.5]
    w = barrier_constrain([0.7, 0.1, 0.1, 0.1], groups=[[0, 1], [2, 3]], group_caps=caps, tau=1e-6)
    assert abs(w.sum() - 1.0) < 1e-12
    assert np.all(w > 0) and np.all(w < 1)
    assert w[:2].sum() < caps[0] and w[2:].sum() < caps[1]


@pytest.mark.parametrize("groups, caps, match", [
    ([[0], [1]], [0.4], "group_caps"),
    ([[0, 1], [1, 2]], [0.6, 0.6], "disjoint"),
    ([[0, 5]], [0.6], r"\[0, 3\)"),
    ([[0, 1]], None, "together"),
])
def test_malformed_groups_raise(groups, caps, match):
    with pytest.raises(ValueError, match=match):
        barrier_constrain(np.full(3, 1 / 3), groups=groups, group_caps=caps)


@pytest.mark.parametrize("tau", [0.0, -1e-4, np.nan])
def test_nonpositive_tau_raises(tau):
    with pytest.raises(ValueError, match="tau"):
        barrier_constrain([0.5, 0.5], tau=tau)


def test_wrappers_reject_an_infeasible_cap():
    X = np.random.default_rng(0).standard_normal((30, 2)) * 0.01
    with pytest.raises(ValueError):
        BoxConstrained(EqualWeight(), groups=[[0, 1]], group_caps=[0.9]).fit(X)
    s = StreamingBoxConstrained(StreamingEqualWeight(min_obs=1),
                                groups={"a": "g", "b": "g"}, group_caps={"g": 0.9})
    with pytest.raises(ValueError):
        s.learn_one({"a": 0.01, "b": -0.01})
