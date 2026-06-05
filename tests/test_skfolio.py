import numpy as np
import pytest

from allocation import ThurstonePortfolio


def _returns(n_obs=400, n=4, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n_obs, 1))
    return f * rng.uniform(0.3, 0.9, n) + 0.5 * rng.standard_normal((n_obs, n))


def test_score_is_finite_float():
    X = _returns()
    s = ThurstonePortfolio().fit(X).score(X)
    assert isinstance(s, float) and np.isfinite(s)


def test_to_portfolio_requires_skfolio():
    X = _returns()
    est = ThurstonePortfolio().fit(X)
    skfolio = pytest.importorskip("skfolio")  # skip if not installed
    pf = est.to_portfolio(X)
    assert isinstance(pf, skfolio.Portfolio)
    assert np.allclose(np.asarray(pf.weights), est.weights_)
