"""The tilting round trip, as a new user meets it."""
import numpy as np
import pytest

from allocation import (
    abilities_from_weights,
    blend_correlation,
    tilt_weights,
    weights_from_abilities,
)


def test_round_trip_is_the_advertised_one_liner():
    w = np.array([0.4, 0.25, 0.2, 0.15])
    assert np.abs(weights_from_abilities(abilities_from_weights(w)) - w).max() < 1e-8


def test_abilities_are_mean_zero_and_smaller_is_stronger():
    w = np.array([0.5, 0.3, 0.15, 0.05])
    a = abilities_from_weights(w)
    assert abs(a.mean()) < 1e-12
    assert np.all(np.diff(a) > 0)          # weights descending -> abilities ascending


def test_weights_from_abilities_is_shift_invariant():
    a = np.array([-0.4, 0.1, 0.3])
    assert np.abs(weights_from_abilities(a) - weights_from_abilities(a + 7.0)).max() < 1e-12


def test_zero_weights_do_not_raise_and_stay_finite():
    a = abilities_from_weights(np.array([0.5, 0.3, 0.2, 0.0]))
    assert np.isfinite(a).all() and a[3] == a.max()


@pytest.mark.parametrize("sampler", ["gaussian", "student_t"])
def test_tilt_recovers_the_benchmark_at_phi_zero(sampler):
    """Only up to the race's Monte Carlo error, which is the honest contract."""
    w = np.array([0.4, 0.25, 0.2, 0.15])
    C = np.eye(4)
    out = tilt_weights(w, C, phi=0.0, sampler=sampler, n_paths=1 << 16)
    assert np.abs(out - w).max() < 0.02
    assert abs(out.sum() - 1.0) < 1e-12


def test_tilt_moves_the_benchmark_and_survives_a_singular_covariance():
    w = np.full(4, 0.25)
    C = np.array([[1, .6, .2, .1], [.6, 1, .3, .1], [.2, .3, 1, .4], [.1, .1, .4, 1.0]])
    assert np.abs(tilt_weights(w, C, phi=1.0) - w).sum() > 1e-3
    out = tilt_weights(w, np.ones((4, 4)), phi=0.5)      # rank one, uninvertible
    assert np.isfinite(out).all() and abs(out.sum() - 1.0) < 1e-12


def test_blend_correlation_is_the_dial():
    C = np.array([[1.0, 0.8], [0.8, 1.0]])
    assert np.abs(blend_correlation(np.eye(2), C, 0.0) - np.eye(2)).max() < 1e-12
    assert np.abs(blend_correlation(np.eye(2), C, 1.0) - C).max() < 1e-12


def test_phi_outside_the_unit_interval_raises():
    with pytest.raises(ValueError):
        tilt_weights(np.full(3, 1 / 3), np.eye(3), phi=1.5)
