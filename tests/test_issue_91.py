"""#91: supplied factor metadata must be over assets, and fit must return n weights.

A transposed (k, n) loadings matrix with a length-k idiosyncratic vector is
conformable for the Woodbury algebra, so fit() used to succeed with k weights.
"""

import numpy as np
import pytest

from allocation import FactorMaximumDiversification, FactorMinimumVariance
from allocation.base import BaseOnlinePortfolio
from allocation.factor import factor_min_variance_weights

FACTOR = (FactorMinimumVariance, FactorMaximumDiversification)
B = np.array([[0.2, 0.0], [0.1, 0.3], [-0.2, 0.1]])  # (n=3, k=2)
PSI = np.array([0.5, 0.4, 0.6])
X = np.arange(12.0).reshape(4, 3)


class _FactorCov:
    def __init__(self, loadings, idiosyncratic):
        self.loadings = loadings
        self.idiosyncratic = idiosyncratic

    def fit(self, X):
        self.covariance_ = B @ B.T + np.diag(PSI)
        self.loadings_ = self.loadings
        self.idiosyncratic_ = self.idiosyncratic
        return self

    def partial_fit(self, X):
        return self


@pytest.mark.parametrize("cls", FACTOR)
@pytest.mark.parametrize("loadings, idio", [
    (B.T, np.array([0.5, 0.4])),          # transposed model, variance per factor
    (B, np.array([0.5, 0.4])),            # wrong-length idiosyncratic variances
    (B[:, 0], PSI),                       # 1-d loadings
    (B, np.array([0.5, -0.4, 0.6])),      # negative idiosyncratic variance
    (B, np.array([0.5, np.nan, 0.6])),    # non-finite
], ids=["transposed", "short_psi", "1d_loadings", "negative_psi", "nan_psi"])
def test_malformed_factor_metadata_raises_at_fit(cls, loadings, idio):
    with pytest.raises(ValueError):
        cls(covariance_estimator=_FactorCov(loadings, idio)).fit(X)


@pytest.mark.parametrize("cls", FACTOR)
def test_valid_supplied_model_returns_one_weight_per_asset(cls):
    est = cls(covariance_estimator=_FactorCov(B, PSI)).fit(X)
    assert est.weights_.shape == (3,)
    assert np.isclose(est.weights_.sum(), 1.0)
    assert est.predict(X).shape == (4,)
    if cls is FactorMinimumVariance:
        np.testing.assert_allclose(est.weights_, factor_min_variance_weights(B, PSI))


class _WrongLength(BaseOnlinePortfolio):
    def _cold_start(self, cov):
        self._weights = np.full(cov.shape[0] - 1, 1.0 / (cov.shape[0] - 1))

    def _online_update(self, cov):
        self._cold_start(cov)


class _NonFinite(BaseOnlinePortfolio):
    def _cold_start(self, cov):
        self._weights = np.full(cov.shape[0], np.nan)

    def _online_update(self, cov):
        self._cold_start(cov)


@pytest.mark.parametrize("cls", [_WrongLength, _NonFinite])
def test_base_fit_rejects_malformed_weights(cls):
    with pytest.raises(ValueError):
        cls().fit(np.random.default_rng(0).standard_normal((20, 4)))
