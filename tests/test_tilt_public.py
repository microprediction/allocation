"""The tilting round trip, as a new user meets it."""
import warnings

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


@pytest.mark.parametrize("n,alpha", [(3, 1.0), (3, 0.2), (4, 1.0), (4, 0.2),
                                     (6, 0.15), (10, 0.2), (25, 1.0), (60, 1.0)])
def test_round_trip_holds_as_a_property_not_just_for_one_vector(n, alpha):
    """The version of this test that shipped asserted a single hardcoded
    vector, [0.4, 0.25, 0.2, 0.15], which round-trips to 2.7e-09 and passed
    every run. On random draws of the same size, 12 percent failed the same
    bound with a worst case of 2.4e-03, and at three assets it was half, with
    the two largest names transposed. The inverse was not an inverse and the
    test could not see it, because it pinned an example rather than the claim.

    Anything that does not converge must warn, so nothing is silently wrong.
    """
    rng = np.random.default_rng(hash((n, alpha)) % (2 ** 32))
    for _ in range(40):
        w = rng.dirichlet(np.full(n, alpha))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            err = np.abs(weights_from_abilities(abilities_from_weights(w)) - w).max()
        warned = any(issubclass(c.category, RuntimeWarning) for c in caught)
        assert err < 1e-6 or warned, (
            f"round trip off by {err:.2e} on {w} with no warning")


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


def test_gaussian_tilt_recovers_the_benchmark_and_improves_with_paths():
    """The Monte Carlo error should fall like 1/sqrt(paths), not sit on a floor."""
    w = np.array([0.4, 0.25, 0.2, 0.15])
    err = {m: np.abs(tilt_weights(w, np.eye(4), phi=0.0, n_paths=m) - w).max()
           for m in (1 << 14, 1 << 18)}
    assert err[1 << 18] < 0.5 * err[1 << 14]
    assert err[1 << 18] < 3e-3


def test_student_t_tilt_does_not_recover_the_benchmark():
    """Deliberate: abilities are calibrated so a GAUSSIAN race reproduces the
    benchmark, and a t race at those abilities is a different law rather than a
    noisy version of the same one. The offset does not shrink with paths, so a
    caller must read t results against the t race at phi=0, not the benchmark.
    """
    w = np.array([0.4, 0.25, 0.2, 0.15])
    big = np.abs(tilt_weights(w, np.eye(4), phi=0.0, sampler="student_t",
                              n_paths=1 << 18) - w).max()
    assert big > 5e-3
    assert abs(tilt_weights(w, np.eye(4), phi=0.0, sampler="student_t").sum() - 1.0) < 1e-12


def test_t_race_has_unit_variance():
    """Without the renormalisation the t race carries variance nu/(nu-2), while
    the abilities were calibrated against a unit-variance race."""
    from allocation._thurstone.transport import _t_scale
    rng = np.random.default_rng(0)
    nu, M = 7.0, 200_000
    X = rng.normal(size=(M, 3)) / _t_scale(rng.chisquare(nu, M), nu)
    assert abs(X.var(0).mean() - 1.0) < 0.02


def test_rejects_weights_it_cannot_invert():
    for bad in (np.array([0.4, 0.3, np.nan, 0.1]),
                np.array([0.6, 0.6, -0.2, 0.0]),
                np.zeros(4)):
        with pytest.raises(ValueError):
            tilt_weights(bad, np.eye(4))


def test_blend_correlation_interpolates_in_the_interior():
    """Both arguments must be converted to correlations. Converting only the
    second leaves both endpoints right and the interior wrong."""
    A = np.array([[4.0, 1.2], [1.2, 1.0]])      # correlation 0.6
    B = np.array([[1.0, 0.8], [0.8, 1.0]])      # correlation 0.8
    for phi in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert abs(blend_correlation(A, B, phi)[0, 1] - (0.6 + 0.2 * phi)) < 1e-9


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


def test_exact_tilt_reproduces_the_benchmark_to_machine_precision():
    """Quadrature, not simulation. The whole seed-ensemble apparatus exists
    because the transport simulates a race that winning evaluates exactly, and
    at phi = 0 the difference is nine orders of magnitude."""
    rng = np.random.default_rng(1)
    n = 120
    V = rng.normal(0, 0.4, (n, 3))
    C = V @ V.T + np.diag(np.maximum(1 - (V ** 2).sum(1), 0.05))
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    w = rng.dirichlet(np.full(n, 3.0))
    assert np.abs(tilt_weights(w, C, phi=0.0, sampler="exact") - w).sum() < 1e-8


def test_exact_and_simulated_agree_on_the_tilt_itself():
    """They should differ only by the simulation's noise, so the exact answer
    must sit near the simulated one at an interior phi."""
    rng = np.random.default_rng(1)
    n = 120
    V = rng.normal(0, 0.4, (n, 3))
    C = V @ V.T + np.diag(np.maximum(1 - (V ** 2).sum(1), 0.05))
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    w = rng.dirichlet(np.full(n, 3.0))
    e = tilt_weights(w, C, phi=1.0, sampler="exact")
    g = tilt_weights(w, C, phi=1.0, sampler="gaussian", n_paths=1 << 16)
    assert np.abs(np.abs(e - w).sum() - np.abs(g - w).sum()) < 0.05


def test_exact_tilt_needs_no_paths_and_no_memory():
    """The point of it: a universe where the seed ensemble would not fit."""
    rng = np.random.default_rng(2)
    n = 3000
    V = rng.normal(0, 0.4, (n, 2))
    D = np.maximum(1 - (V ** 2).sum(1), 0.05)
    s = np.sqrt((V ** 2).sum(1) + D)
    V, D = V / s[:, None], D / s ** 2
    C = V @ V.T + np.diag(D)          # a real covariance, but never inverted
    w = rng.dirichlet(np.full(n, 3.0))
    out = tilt_weights(w, C, phi=0.0, sampler="exact")
    assert np.abs(out - w).sum() < 1e-6
