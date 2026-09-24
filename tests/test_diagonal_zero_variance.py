"""diagonal_portfolio at and near zero variance (issue #70)."""
import numpy as np
import pytest

from allocation._thurstone.diagonal import diagonal_portfolio


def test_riskless_asset_takes_the_allocation_and_is_continuous_at_zero():
    near = diagonal_portfolio(np.diag([1e-12, 1.0]))
    exact = diagonal_portfolio(np.diag([0.0, 1.0]))
    assert np.allclose(exact, [1.0, 0.0])
    assert np.allclose(near, exact, atol=1e-9)
    # portfolio variance does not jump from ~0 to 1 at zero
    S = np.diag([0.0, 1.0])
    assert exact @ S @ exact == 0.0


def test_several_riskless_assets_split_equally_and_others_get_nothing():
    w = diagonal_portfolio(np.diag([0.0, 2.0, 0.0]))
    assert np.allclose(w, [0.5, 0.0, 0.5])


def test_all_zero_variance_is_equal_weight():
    assert np.allclose(diagonal_portfolio(np.zeros((3, 3))), [1 / 3] * 3)


def test_negative_variance_is_rejected():
    with pytest.raises(ValueError, match="negative variance"):
        diagonal_portfolio(np.diag([-1.0, 1.0]))
