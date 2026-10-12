"""Issue #58: one-factor calibration must not silently return a wrong benchmark.

With equal loadings near the 0.99 cap, the winning inverse shipped at the
declared floor (1.5.0) fails to converge for an 80/20 target, warns, and hands
back an ability gap of 1.28 whose forward race is 100/0. Calibration now checks
its own round trip: it either reproduces the target or raises.
"""

import numpy as np
import pytest
from statistics import NormalDist

import allocation._thurstone.calibrate as cal
from allocation._thurstone.calibrate import calibrate_one_factor, winprobs_one_factor


def test_high_loading_pair_is_exact_or_refused():
    for b in [0.8, 0.95, 0.99, 0.999]:
        try:
            a = calibrate_one_factor([0.8, 0.2], [b, b])
        except RuntimeError:
            continue  # an older winning that cannot invert this is refused loudly
        exact = np.sqrt(2 * (1 - b * b)) * NormalDist().inv_cdf(0.8)
        assert abs((a[1] - a[0]) - exact) < 1e-3
        assert np.allclose(winprobs_one_factor(a, [b, b]), [0.8, 0.2], atol=1e-4)


def test_nonconverged_inverse_raises(monkeypatch):
    # stand in for winning 1.5.0 at b=0.99: the last iterate, gap 1.2755
    monkeypatch.setattr(cal.winning, "calibrate_abilities",
                        lambda *args, **kw: np.array([-0.63775, 0.63775]))
    with pytest.raises(RuntimeError, match="did not reproduce the target"):
        calibrate_one_factor([0.8, 0.2], [0.99, 0.99])


def test_round_trip_holds_on_small_nonuniform_universes():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(2, 8))
        w = rng.dirichlet(np.ones(n))
        b = rng.uniform(0.3, 0.95, n)
        a = calibrate_one_factor(w, b)
        assert np.max(np.abs(winprobs_one_factor(a, b) - w)) < 1e-3
