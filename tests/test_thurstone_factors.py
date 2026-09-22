import numpy as np

from allocation import ThurstonePortfolio
from allocation._thurstone.covariance import factor_decompose
from allocation._thurstone.transport import transport_weights, transport_weights_lowrank


def _factor_returns(n_obs=600, n=12, k=3, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n_obs, k))
    load = rng.uniform(-0.7, 0.9, size=(n, k))
    return f @ load.T + 0.5 * rng.standard_normal((n_obs, n))


def test_lowrank_transport_matches_dense_winprobs():
    # on a k-factor correlation C = B B^T + diag(d), the low-rank race and the
    # dense race estimate the same win probabilities (agree up to MC noise).
    rng = np.random.default_rng(0)
    n, k, M = 10, 3, 400_000
    B = rng.standard_normal((n, k)) * 0.35
    d = 1.0 - (B**2).sum(1)
    assert np.all(d > 0)
    C = B @ B.T + np.diag(d)
    a = rng.standard_normal(n) * 0.5
    w_dense = transport_weights(a, C, rng.standard_normal((M, n)))
    w_low = transport_weights_lowrank(
        a, B, d, rng.standard_normal((M, k)), rng.standard_normal((M, n))
    )
    assert np.max(np.abs(w_dense - w_low)) < 0.01


def test_factor_decompose_reconstructs_unit_diagonal():
    C = np.corrcoef(_factor_returns(n=15, k=3), rowvar=False)
    B, d = factor_decompose(C, k=3)
    approx = B @ B.T + np.diag(d)
    assert B.shape == (15, 3)
    assert np.allclose(np.diag(approx), 1.0, atol=1e-6)  # valid correlation diagonal
    # captures the dominant structure of a (near) rank-3 correlation
    assert np.linalg.norm(approx - C) < 0.5 * np.linalg.norm(C - np.eye(15))


def test_factor_mode_long_only_simplex_and_smooth():
    X = _factor_returns(n=20, k=3)
    est = ThurstonePortfolio(calib="market", factors=3, n_paths=1 << 12).fit(X)
    w = est.weights_
    assert np.all(w >= -1e-9) and abs(float(w.sum()) - 1.0) < 1e-6
    before = w.copy()
    est.partial_fit(_factor_returns(n_obs=5, n=20, k=3, seed=1))
    assert np.max(np.abs(est.weights_ - before)) < 0.15  # smooth update


def test_factor_mode_close_to_dense_on_factor_data():
    # data that is genuinely 3-factor: the 3-factor tilt should track the dense
    # tilt closely (factor-truncation error small), up to MC noise.
    X = _factor_returns(n=16, k=3, seed=2)
    dense = ThurstonePortfolio(calib="market", n_paths=1 << 14, seed=7).fit(X).weights_
    fac = ThurstonePortfolio(calib="market", factors=3, n_paths=1 << 14, seed=7).fit(X).weights_
    assert np.max(np.abs(dense - fac)) < 0.07


def test_factor_mode_runs_on_large_universe():
    # the whole point: a few hundred names is fine via the O(M n k) transport
    # (diagonal calibration keeps this about the transport, not the calibrator).
    X = _factor_returns(n_obs=200, n=300, k=4, seed=3)
    w = ThurstonePortfolio(calib="diagonal", factors=4, n_paths=1 << 12).fit(X).weights_
    assert w.shape == (300,)
    assert abs(float(w.sum()) - 1.0) < 1e-6 and np.all(w >= -1e-9)


def test_low_rank_tilt_preserves_the_zero_confidence_anchor():
    """At phi=0 the tilt must return the benchmark, low-rank or not.

    Issue #57. The low-rank path used to blend the correlations and then factor
    the blend. At phi=0 that means factoring the calibration reference, and for
    calib="diagonal" the reference is the identity, whose eigendecomposition has
    no distinguished leading direction. Truncating it to k factors and restoring
    the diagonal invented correlation out of an arbitrary basis: an equal target
    came back spread from 0.110 to 0.149 instead of 0.125.

    The blend is formed in factor space now, where two k-factor correlations
    combine exactly into one of rank 2k, so both endpoints are reproduced with
    no eigendecomposition of the blend anywhere.
    """
    X = np.random.default_rng(42).normal(size=(120, 8))
    for calib in ("diagonal", "market"):
        dense = ThurstonePortfolio(target="equal", calib=calib, phi=0,
                                   factors=None, n_paths=2 ** 18).fit(X)
        low = ThurstonePortfolio(target="equal", calib=calib, phi=0,
                                 factors=3, n_paths=2 ** 18).fit(X)
        w = np.asarray(low.weights_, float)
        assert np.abs(w - 0.125).max() < 5e-3, (calib, w)
        assert (np.abs(w - 0.125).max()
                <= np.abs(np.asarray(dense.weights_, float) - 0.125).max() + 2e-3)


def test_full_rank_tilt_matches_the_dense_race():
    """At k = n the low-rank path is not an approximation and must agree.

    This separates the anchor from rank truncation. A gap at k < n is the
    documented approximation; a gap at k = n would be a defect.
    """
    Y = (np.random.default_rng(7).normal(size=(300, 8))
         @ np.linalg.cholesky(np.eye(8) * 0.4 + 0.6).T)
    dense = ThurstonePortfolio(target="equal", calib="diagonal", phi=1.0,
                               factors=None, n_paths=2 ** 18).fit(Y).weights_
    full = ThurstonePortfolio(target="equal", calib="diagonal", phi=1.0,
                              factors=8, n_paths=2 ** 18).fit(Y).weights_
    assert np.abs(np.asarray(full, float) - np.asarray(dense, float)).max() < 5e-3
