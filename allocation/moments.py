"""Online covariance for the portfolio estimators.

A light, numpy-only EWMA covariance is the default so the package has no heavy
dependencies. Any external online estimator that exposes a ``covariance_``
attribute and a ``partial_fit`` method (e.g. the ``precise`` covariance skaters)
can be plugged in instead -- the portfolio estimators only ever read
``covariance_``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["EwmaCovariance", "DownsideSemicovariance"]


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

    @property
    def mean_(self) -> np.ndarray:
        """EWMA mean of the observed returns (itself a smooth path)."""
        if self._mean is None:
            raise ValueError("EwmaCovariance has not seen any data yet.")
        return self._mean


class DownsideSemicovariance:
    """Online co-lower-partial-moment (downside semicovariance) estimator.

    Maintains an EWMA estimate of ``E[ d d^T ]`` with
    ``d_i = min(r_i - tau_i, 0)`` -- the downside-only deviations from a threshold
    ``tau`` (the running EWMA mean by default, or a fixed target return). The result
    is PSD by construction, and its correlation is the *downside* correlation, which
    can differ sharply from the full correlation when assets crash together but rally
    independently. Drop-in for the portfolio estimators
    (``ThurstonePortfolio(covariance_estimator=DownsideSemicovariance())``): driving
    the race with it makes the de-duplication key on genuine joint-tail co-movement
    rather than average covariance.

    Parameters
    ----------
    halflife : float, default 60.0
        EWMA halflife for both the threshold mean and the semicovariance.
    threshold : float or None, default None
        ``None`` uses the running EWMA mean as the per-asset threshold; a float uses
        that constant target return (e.g. ``0.0``) for every asset.
    """

    def __init__(self, halflife: float = 60.0, threshold: float | None = None):
        self.halflife = halflife
        self.threshold = threshold
        self._mean: np.ndarray | None = None
        self._semicov: np.ndarray | None = None
        self.n_samples_ = 0

    def _rows(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X[None, :] if X.ndim == 1 else X

    def partial_fit(self, X, y=None) -> "DownsideSemicovariance":
        alpha = _halflife_to_alpha(self.halflife)
        for x in self._rows(X):
            if self._mean is None:
                n = len(x)
                self._mean = x.copy()
                self._semicov = np.zeros((n, n), dtype=float)
            else:
                self._mean = (1 - alpha) * self._mean + alpha * x
            tau = self._mean if self.threshold is None else self.threshold
            d = np.minimum(x - tau, 0.0)  # downside-only deviations
            self._semicov = (1 - alpha) * self._semicov + alpha * np.outer(d, d)
            self.n_samples_ += 1
        return self

    def fit(self, X, y=None) -> "DownsideSemicovariance":
        self._mean = None
        self._semicov = None
        self.n_samples_ = 0
        return self.partial_fit(X)

    @property
    def covariance_(self) -> np.ndarray:
        if self._semicov is None:
            raise ValueError("DownsideSemicovariance has not seen any data yet.")
        S = 0.5 * (self._semicov + self._semicov.T)
        # floor the diagonal so the implied correlation is well-defined even for an
        # asset that has not yet printed a downside move.
        d = np.diag(S)
        floor = max(float(d.max()) * 1e-8, 1e-300)
        if np.any(d < floor):
            S = S + np.diag(np.maximum(floor - d, 0.0))
        return S

    @property
    def mean_(self) -> np.ndarray:
        if self._mean is None:
            raise ValueError("DownsideSemicovariance has not seen any data yet.")
        return self._mean
