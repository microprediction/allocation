"""Invalid solver controls must not certify a merely feasible portfolio (#142)."""
import numpy as np
import pytest

from allocation.constraints import barrier_constrain


@pytest.mark.parametrize('tol', [True, False, np.bool_(True), np.nan, np.inf,
                                 -np.inf, 0.0, -1e-10, '1e-10', 1e-10j,
                                 [1e-10], np.array(1e-10)])
@pytest.mark.parametrize('point', [False, True])
def test_invalid_tolerance_rejected_even_for_unique_point(tol, point):
    with pytest.raises(ValueError, match='tol'):
        barrier_constrain([.8, .2], upper=.5 if point else 1., tol=tol)


@pytest.mark.parametrize('max_iter', [True, False, np.bool_(True), 0, -1, 1.5,
                                      100.0, np.nan, np.inf, '100', [100],
                                      np.array(100)])
@pytest.mark.parametrize('point', [False, True])
def test_invalid_iteration_count_rejected_even_for_unique_point(max_iter, point):
    with pytest.raises(ValueError, match='max_iter'):
        barrier_constrain([.8, .2], upper=.5 if point else 1., max_iter=max_iter)


def test_default_result_satisfies_stationarity_and_numpy_controls_match():
    kw = dict(target=[10., -5., 2., -1.], lower=[-.2, -.1, 0., 0.],
              upper=[.8, .7, .6, .5], groups=[[0, 1]], group_caps=[.6], tau=.01)
    w = barrier_constrain(**kw)
    np.testing.assert_array_equal(w, barrier_constrain(**kw, max_iter=np.int64(100),
                                                      tol=np.float64(1e-10)))
    g = w - kw['target'] - kw['tau'] * (1 / (w - kw['lower']) - 1 / (kw['upper'] - w))
    g[:2] += kw['tau'] / (.6 - w[:2].sum())
    assert np.ptp(g) < 1e-7
    assert abs(w.sum() - 1.) < 1e-12


def test_positive_but_insufficient_budget_still_reports_nonconvergence():
    with pytest.raises(RuntimeError, match='did not converge'):
        barrier_constrain([.9, .1], max_iter=1)
