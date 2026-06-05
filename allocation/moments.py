"""Online covariance for the portfolio estimators.

A light, numpy-only EWMA covariance is the default so the package has no heavy
dependencies. Any external online estimator that exposes a ``covariance_``
attribute and a ``partial_fit`` method (e.g. the ``precise`` covariance skaters)
can be plugged in instead -- the portfolio estimators only ever read
``covariance_``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["EwmaCovariance"]


def _halflife_to_alpha(halflife: float) -> float:
    return 1.0 - 0.5 ** (1.0 / max(float(halflife), 1e-9))


class EwmaCovariance:
    """Exponentially-weighted online mean/covariance estimator.

    Mirrors the minimal surface the portfolio estimators rely on:
    ``partial_fit(X)`` / ``fit(X)`` and the fitted ``covariance_``.
    """

    def __init__(self, halflife: float = 60.0):
        self.halflife = halflife
        self._mean: np.ndarray | None = None
        self._cov: np.ndarray | None = None
        self.n_samples_ = 0

    def _rows(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X[None, :] if X.ndim == 1 else X

    def partial_fit(self, X, y=None) -> "EwmaCovariance":
        alpha = _halflife_to_alpha(self.halflife)
        for x in self._rows(X):
            if self._mean is None:
                n = len(x)
                self._mean = x.copy()
                self._cov = np.zeros((n, n), dtype=float)
            else:
                self._mean = (1 - alpha) * self._mean + alpha * x
            dev = x - self._mean
            self._cov = (1 - alpha) * self._cov + alpha * np.outer(dev, dev)
            self.n_samples_ += 1
        return self

    def fit(self, X, y=None) -> "EwmaCovariance":
        self._mean = None
        self._cov = None
        self.n_samples_ = 0
        return self.partial_fit(X)

    @property
    def covariance_(self) -> np.ndarray:
        if self._cov is None:
            raise ValueError("EwmaCovariance has not seen any data yet.")
        return 0.5 * (self._cov + self._cov.T)
