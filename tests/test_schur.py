import numpy as np

from allocation import SchurComplementary
from allocation._schur.coupling import compute_weights
from allocation._schur.seriation import seriate


def _returns(n_obs=600, n=8, seed=0):
    rng = np.random.default_rng(seed)
    # two correlated blocks -> non-trivial seriation + cross-block coupling
    f = rng.standard_normal((n_obs, 2))
    load = np.zeros((n, 2))
    load[: n // 2, 0] = rng.uniform(0.4, 0.9, n // 2)
    load[n // 2 :, 1] = rng.uniform(0.4, 0.9, n - n // 2)
    return f @ load.T + 0.5 * rng.standard_normal((n_obs, n))


def _cov(X):
    return np.cov(X, rowvar=False)


def test_compute_weights_long_only_simplex():
    cov = _cov(_returns())
    order, _ = seriate(cov)
    for gamma in (0.0, 0.5, 1.0):
        w = compute_weights(order, cov, gamma)
        assert np.all(w >= -1e-9)
        assert abs(float(w.sum()) - 1.0) < 1e-9


def test_seriate_is_a_permutation():
    cov = _cov(_returns())
    order, v = seriate(cov)
    assert sorted(order.tolist()) == list(range(len(cov)))
    assert len(v) == len(cov)


def test_gamma_zero_is_hrp_ignores_cross_block():
    # At gamma=0 the augmentation is the identity, so weights equal the plain HRP
    # recursive-bisection allocation (no cross-block info).
    cov = _cov(_returns())
    order, _ = seriate(cov)
    w0 = compute_weights(order, cov, 0.0)
    # block-diagonalising the covariance must not change the gamma=0 weights
    cov_bd = cov.copy()
    half = len(cov) // 2
    o = order
    cov_bd[np.ix_(o[:half], o[half:])] = 0.0
    cov_bd[np.ix_(o[half:], o[:half])] = 0.0
    w0_bd = compute_weights(order, cov_bd, 0.0)
    assert np.allclose(w0, w0_bd, atol=1e-8)


def test_estimator_fit_and_partial_fit():
    X = _returns()
    est = SchurComplementary(gamma=0.5).fit(X)
    w = est.weights_
    assert np.all(w >= -1e-9) and abs(float(w.sum()) - 1.0) < 1e-6
    assert sorted(est.order_.tolist()) == list(range(X.shape[1]))
    w_before = w.copy()
    est.partial_fit(_returns(n_obs=5, seed=1))
    assert np.max(np.abs(est.weights_ - w_before)) < 0.2  # smooth update


def test_gamma_out_of_range_raises():
    X = _returns()
    try:
        SchurComplementary(gamma=1.5).fit(X)
    except ValueError:
        return
    raise AssertionError("expected ValueError for gamma out of [0,1]")
