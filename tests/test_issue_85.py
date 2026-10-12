"""#85: a detected-singular covariance must take the least-squares path.

``np.linalg.solve`` on a numerically rank-deficient matrix need not raise, so
the old try/except fallback could return a roundoff-dependent answer that
changed when the assets were merely relabeled.
"""

import numpy as np
import pytest

from allocation import MaximumDecorrelation, MaximumDiversification, MinimumVariance
from allocation._thurstone.covariance import cov_to_corr

ESTIMATORS = (MinimumVariance, MaximumDiversification, MaximumDecorrelation)


class _Fixed:
    def __init__(self, covariance):
        self.covariance_ = covariance

    def fit(self, X):
        return self


def _low_rank():
    X = np.array([
        [1., 2., 3., 4., 5., 6.],
        [-1., 1., 0., 2., -2., 3.],
        [2., 0., -1., 1., 3., -2.],
    ])
    return X.T @ X  # PSD, rank 3 in dimension 6


def _sample_cov(T=4, n=9, seed=0):
    R = np.random.default_rng(seed).standard_normal((T, n))
    return np.cov(R, rowvar=False)  # rank T - 1 < n


def _fit(cls, cov, **kw):
    return cls(covariance_estimator=_Fixed(cov), **kw).fit(np.zeros((1, cov.shape[0])))


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("cov", [_low_rank(), _sample_cov()], ids=["low_rank", "T_lt_n"])
def test_singular_weights_do_not_depend_on_asset_labels(cls, cov):
    n = cov.shape[0]
    rng = np.random.default_rng(1)
    a = _fit(cls, cov)
    assert a.singular_
    for _ in range(5):
        p = rng.permutation(n)
        b = _fit(cls, cov[np.ix_(p, p)])
        assert b.singular_
        np.testing.assert_allclose(b.weights_[np.argsort(p)], a.weights_, atol=1e-9)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_singular_weights_match_pseudoinverse(cls):
    cov = _low_rank()
    n = cov.shape[0]
    if cls is MinimumVariance:
        A, rhs = cov, np.ones(n)
    elif cls is MaximumDiversification:
        A, rhs = cov, np.sqrt(np.diag(cov))
    else:
        A, rhs = cov_to_corr(cov), np.ones(n)
    z = np.linalg.pinv(A, hermitian=True) @ rhs
    np.testing.assert_allclose(_fit(cls, cov).weights_, z / z.sum(), atol=1e-9)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_strict_still_raises_and_shrinkage_restores_full_rank(cls):
    cov = _low_rank()
    with pytest.raises(np.linalg.LinAlgError):
        _fit(cls, cov, strict=True)
    est = _fit(cls, cov, shrinkage=0.3, strict=True)
    assert not est.singular_
    assert np.isclose(est.weights_.sum(), 1.0)
