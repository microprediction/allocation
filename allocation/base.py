"""Base class for online portfolio estimators.

``BaseOnlinePortfolio`` follows scikit-learn / skfolio conventions: hyperparameters
in ``__init__``, ``fit(X)`` / ``partial_fit(X)`` taking asset returns ``X`` and
storing weights in ``weights_``, plus a ``predict`` that returns portfolio
returns. Subclasses implement two hooks:

* ``_cold_start(cov)`` -- compute weights from scratch given a covariance, and
  initialise any persistent state (e.g. a fixed seed ensemble);
* ``_online_update(cov)`` -- update weights for a new covariance, reusing that
  state (this is where the smooth, low-turnover transport happens).

Covariance is produced by a pluggable estimator (default :class:`EwmaCovariance`);
anything exposing ``partial_fit`` and ``covariance_`` works, including the
``precise`` skaters.
"""

from __future__ import annotations

import numpy as np

from .moments import EwmaCovariance

__all__ = ["BaseOnlinePortfolio"]


class NotFittedError(ValueError):
    pass


class BaseOnlinePortfolio:
    # subclasses set hyperparameters in their own __init__ then call super().__init__()
    def __init__(self, covariance_estimator=None, halflife: float = 60.0):
        self.covariance_estimator = covariance_estimator
        self.halflife = halflife
        self._cov_estimator = None
        self._weights: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self._n_updates = 0

    # ------------------------------------------------------------------ hooks
    def _cold_start(self, cov: np.ndarray) -> None:
        raise NotImplementedError

    def _online_update(self, cov: np.ndarray) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------- internals
    def _new_cov_estimator(self):
        if self.covariance_estimator is None:
            return EwmaCovariance(halflife=self.halflife)
        # a pluggable estimator instance (skfolio.moments, precise skater, ...)
        est = self.covariance_estimator
        if hasattr(est, "fit") or hasattr(est, "partial_fit"):
            return est
        raise TypeError(
            "covariance_estimator must be None or expose fit/partial_fit and covariance_"
        )

    @staticmethod
    def _as_2d(X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X[None, :] if X.ndim == 1 else X

    def _feed(self, X) -> np.ndarray:
        self._cov_estimator.partial_fit(X)
        return np.asarray(self._cov_estimator.covariance_, dtype=float)

    # -------------------------------------------------------------- fitting
    def fit(self, X, y=None):
        X = self._as_2d(X)
        self.n_features_in_ = X.shape[1]
        self._cov_estimator = self._new_cov_estimator()
        if hasattr(self._cov_estimator, "fit"):
            self._cov_estimator.fit(X)
        cov = np.asarray(self._cov_estimator.covariance_, dtype=float)
        self._cold_start(cov)
        self._n_updates = 1
        return self

    def partial_fit(self, X, y=None):
        if self._weights is None:
            return self.fit(X)
        X = self._as_2d(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "number of assets changed; keyed dynamic-universe support is not "
                "wired up yet (see universe.py). Re-fit, or pass a fixed universe."
            )
        cov = self._feed(X)
        self._online_update(cov)
        self._n_updates += 1
        return self

    # --------------------------------------------------- fitted attributes
    @property
    def weights_(self) -> np.ndarray:
        if self._weights is None:
            raise NotFittedError(f"{type(self).__name__} is not fitted; call fit first.")
        return self._weights

    def predict(self, X) -> np.ndarray:
        """Portfolio returns of each row of ``X`` under the fitted weights."""
        return self._as_2d(X) @ self.weights_

    def score(self, X, y=None) -> float:
        """Per-period Sharpe ratio of the portfolio on ``X`` (higher is better).

        Follows skfolio's convention that an optimizer's score is the Sharpe
        ratio of its predicted portfolio, so the estimator works with
        scikit-learn / skfolio model-selection out of the box.
        """
        r = np.asarray(self.predict(X), dtype=float)
        sd = float(np.std(r))
        return float(np.mean(r) / sd) if sd > 0 else 0.0

    def to_portfolio(self, X):
        """Wrap the fitted weights as a skfolio ``Portfolio`` (optional dependency).

        Lets the estimator's output drop into skfolio's metrics, ``Population``,
        and plotting, and is the bridge for contributing the estimator upstream
        (where ``predict`` returns a ``Portfolio`` natively). Requires skfolio.
        """
        try:
            from skfolio import Portfolio
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "to_portfolio requires skfolio (pip install skfolio)"
            ) from exc
        return Portfolio(X=X, weights=self.weights_)

    def get_params(self, deep: bool = True) -> dict:
        import inspect

        names = [p for p in inspect.signature(self.__init__).parameters if p != "self"]
        return {k: getattr(self, k) for k in names}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
