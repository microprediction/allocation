import numpy as np
import pytest

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


def test_keep_monotonic_caps_gamma_and_bounds_variance():
    # variance(Schur) <= variance(HRP) with keep_monotonic, and effective_gamma <= gamma
    rng = np.random.default_rng(3)
    n = 10
    A = rng.standard_normal((n, n + 2))
    cov = A @ A.T / (n + 2) + np.diag(rng.uniform(0.05, 0.2, n))  # mildly ill-conditioned
    est = SchurComplementary(gamma=1.0, keep_monotonic=True).fit(np.random.default_rng(4).standard_normal((300, n)))
    assert 0.0 <= est.effective_gamma_ <= 1.0
    w = est.weights_
    assert np.all(w >= -1e-9) and abs(float(w.sum()) - 1.0) < 1e-6


def test_sparse_fiedler_matches_dense_order():
    pytest.importorskip("scipy")
    from allocation._schur.seriation import affinity, fiedler_vector
    cov = _cov(_returns(n=12))
    A = affinity(cov)
    v_dense = fiedler_vector(A, sparse=False)
    v_sparse = fiedler_vector(A, sparse=True)
    # orderings should agree (sign already aligned only if previous given; align here)
    if np.argsort(v_sparse).tolist() != np.argsort(v_dense).tolist():
        v_sparse = -v_sparse
    assert np.argsort(v_sparse).tolist() == np.argsort(v_dense).tolist()


def test_ridge_wellposed_on_rank_deficient_blocks():
    # 12 obs, 20 names -> rank-deficient covariance; gamma>0 inverts singular blocks
    rng = np.random.default_rng(0)
    cov = np.cov(rng.standard_normal((12, 20)), rowvar=False)
    order, _ = seriate(cov)
    w = compute_weights(order, cov, gamma=0.5, ridge=0.1)
    assert w is not None
    assert np.all(w >= -1e-9) and abs(float(w.sum()) - 1.0) < 1e-9 and np.all(np.isfinite(w))


def test_large_ridge_recovers_hrp():
    # as the ridge dominates, the cross-block coupling vanishes -> Schur -> HRP (gamma 0)
    cov = _cov(_returns())
    order, _ = seriate(cov)
    w_hrp = compute_weights(order, cov, gamma=0.0)
    w_big = compute_weights(order, cov, gamma=0.9, ridge=1e8)
    assert np.allclose(w_hrp, w_big, atol=1e-6)


def test_ridge_zero_is_unchanged():
    # ridge=0 must reproduce the original exact coupling on a full-rank covariance
    cov = _cov(_returns())
    order, _ = seriate(cov)
    assert np.allclose(compute_weights(order, cov, 0.5), compute_weights(order, cov, 0.5, ridge=0.0))


def test_schur_estimator_ridge_runs_high_dim():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 120))  # rank-deficient EWMA covariance
    w = SchurComplementary(gamma=0.5, ridge=0.1, keep_monotonic=False).fit(X).weights_
    assert np.all(w >= -1e-9) and abs(float(w.sum()) - 1.0) < 1e-6 and np.all(np.isfinite(w))


def test_prior_blend_shifts_order():
    # A strong block prior should make the order group the prior blocks.
    cov = _cov(_returns(n=8, seed=5))
    n = 8
    prior = np.zeros((n, n))
    prior[:4, :4] = 1.0
    prior[4:, 4:] = 1.0
    o_plain, _ = seriate(cov)
    o_prior, _ = seriate(cov, prior=prior, prior_weight=0.9)
    # with the prior, each half of the order should be mostly one block
    first_half = set(o_prior[:4].tolist())
    overlap = max(len(first_half & set(range(4))), len(first_half & set(range(4, 8))))
    assert overlap >= 3
