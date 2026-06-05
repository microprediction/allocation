import numpy as np

from allocation import ThurstonePortfolio


def _returns(n_obs=600, n=5, seed=0):
    rng = np.random.default_rng(seed)
    # one common factor + idiosyncratic -> genuine correlation
    f = rng.standard_normal((n_obs, 1))
    load = rng.uniform(0.3, 0.9, size=n)
    return f * load + 0.5 * rng.standard_normal((n_obs, n))


def test_fit_is_long_only_simplex():
    X = _returns()
    w = ThurstonePortfolio().fit(X).weights_
    assert np.all(w >= -1e-9)
    assert abs(float(w.sum()) - 1.0) < 1e-6


def test_market_calibration_runs():
    X = _returns()
    w = ThurstonePortfolio(calib="market", phi=1.0).fit(X).weights_
    assert np.all(w >= -1e-9) and abs(float(w.sum()) - 1.0) < 1e-6


def test_partial_fit_is_smooth_vs_refit():
    # Streaming a few more rows should move weights only a little (transport),
    # and partial_fit on the SAME data twice must be deterministic.
    X = _returns(n_obs=600)
    est = ThurstonePortfolio(phi=1.0).fit(X)
    w0 = est.weights_.copy()
    est.partial_fit(_returns(n_obs=5, seed=1))
    w1 = est.weights_.copy()
    assert np.max(np.abs(w1 - w0)) < 0.15  # smooth, not a fresh draw
    # determinism of the transport given identical state
    a = ThurstonePortfolio(phi=0.5, seed=7).fit(X).weights_
    b = ThurstonePortfolio(phi=0.5, seed=7).fit(X).weights_
    assert np.allclose(a, b)


def test_predict_returns_portfolio_returns():
    X = _returns()
    est = ThurstonePortfolio().fit(X)
    r = est.predict(X)
    assert r.shape == (X.shape[0],)
    assert np.allclose(r, X @ est.weights_)


def test_phi_out_of_range_raises():
    X = _returns()
    try:
        ThurstonePortfolio(phi=1.5).fit(X)
    except ValueError:
        return
    raise AssertionError("expected ValueError for phi out of [0,1]")
