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
        """Update on one row or a whole block.

        The covariance recursion unrolls exactly. After ``T`` rows,

            cov_T = r^T cov_0 + alpha * sum_t r^(T-1-t) dev_t dev_t'

        with ``r = 1 - alpha``, so the block case is one weighted matmul rather
        than ``T`` rank-one outer products. The running mean still needs a scan,
        but that is O(T n) against the covariance's O(T n^2). Which of the two
        dominates depends on the shape: at T=10000, n=50 the scan is most of
        the time and the matmul is a fraction of it, and they cross over near
        n=700. Results agree with the row-at-a-time recursion to about 2e-15,
        which is reassociation rather than a different answer.
        """
        alpha = _halflife_to_alpha(self.halflife)
        rows = self._rows(X)
        if rows.size == 0:
            return self
        r = 1.0 - alpha
        T, n = rows.shape

        fresh = self._mean is None
        if fresh:
            self._mean = rows[0].copy()
            self._cov = np.zeros((n, n), dtype=float)

        devs = np.empty((T, n), dtype=float)
        m = self._mean
        for t in range(T):
            if not (fresh and t == 0):
                m = r * m + alpha * rows[t]
            devs[t] = rows[t] - m
        self._mean = m

        # fold the weights into the deviations in place. The obvious
        # devs.T @ (w[:, None] * devs) allocates a second T x n array, which at
        # T=100000, n=500 is 807 MB of peak against 6 MB for the old row loop.
        # Half-weighting each side is algebraically identical, halves the peak,
        # is faster, and makes the product exactly symmetric.
        devs *= (np.sqrt(alpha) * r ** (0.5 * np.arange(T - 1, -1, -1)))[:, None]
        self._cov = (r ** T) * self._cov + devs.T @ devs
        self.n_samples_ += T
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
        """Same unrolling as :class:`EwmaCovariance`, with a different deviation.

        This class is documented as a drop-in for the covariance estimator, so
        leaving it on the row-at-a-time path would mean anyone taking that
        advice loses the speedup.
        """
        alpha = _halflife_to_alpha(self.halflife)
        rows = self._rows(X)
        if rows.size == 0:
            return self
        r = 1.0 - alpha
        T, n = rows.shape

        fresh = self._mean is None
        if fresh:
            self._mean = rows[0].copy()
            self._semicov = np.zeros((n, n), dtype=float)

        devs = np.empty((T, n), dtype=float)
        m = self._mean
        for t in range(T):
            if not (fresh and t == 0):
                m = r * m + alpha * rows[t]
            tau = m if self.threshold is None else self.threshold
            devs[t] = np.minimum(rows[t] - tau, 0.0)   # downside-only
        self._mean = m

        devs *= (np.sqrt(alpha) * r ** (0.5 * np.arange(T - 1, -1, -1)))[:, None]
        self._semicov = (r ** T) * self._semicov + devs.T @ devs
        self.n_samples_ += T
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
