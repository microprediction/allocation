"""Issue #89: an array target must have exactly one entry per asset.

A singleton target fitted to three assets used to broadcast through the race
and return a plausible equal-weight portfolio, with ``target_`` of shape (1,)
beside ``weights_`` of shape (3,). Other wrong lengths failed later with
low-level broadcasting or matmul errors.
"""

import numpy as np
import pytest

from allocation import ThurstonePortfolio

X = np.random.default_rng(7).normal(size=(200, 3))

BAD = [np.array([1.0]), np.array([0.5, 0.5]), np.full(4, 0.25),
       np.array([[0.2, 0.3, 0.5]])]


@pytest.mark.parametrize("calib", ["diagonal", "market"])
@pytest.mark.parametrize("factors", [None, 2])
@pytest.mark.parametrize("target", BAD, ids=["singleton", "n-1", "n+1", "row"])
def test_wrong_shape_target_is_refused(target, calib, factors):
    est = ThurstonePortfolio(target=target, calib=calib, factors=factors,
                             phi=0.0, n_paths=1024, seed=7)
    with pytest.raises(ValueError, match=r"expected shape \(3,\)"):
        est.fit(X)


@pytest.mark.parametrize("calib", ["diagonal", "market"])
@pytest.mark.parametrize("factors", [None, 2])
def test_correct_length_target_still_fits(calib, factors):
    t = np.array([0.2, 0.3, 0.5])
    est = ThurstonePortfolio(target=list(t), calib=calib, factors=factors,
                             phi=0.0, n_paths=1 << 14, seed=7).fit(X)
    assert est.target_.shape == (3,) and est.weights_.shape == (3,)
    assert np.allclose(est.target_, t)
    assert np.max(np.abs(est.weights_ - t)) < 0.03  # phi=0 reproduces the target
