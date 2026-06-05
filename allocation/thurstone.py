"""Thurstone portfolio: the ability tilt as an online estimator.

Weights are the winning probabilities of a Thurstonian race among assets.
Abilities are calibrated so that the race under a reference correlation
``C_calib`` reproduces a target benchmark; the tilt then re-runs the race under
the estimated correlation ``C_tilt``. ``partial_fit`` transports a fixed seed
ensemble to the updated correlation, so weights move smoothly (low turnover).

See the paper *Thurstone Portfolios: Long-Only Allocation by Inverting Winning
Probabilities* for the theory (feasibility, redundancy consistency, the implied
regularized objective, and the smoothness/turnover bound).
"""

from __future__ import annotations

import numpy as np

from .base import BaseOnlinePortfolio
from ._thurstone.ability import base_density
from ._thurstone.calibrate import calibrate_diagonal, calibrate_one_factor
from ._thurstone.covariance import market_betas, one_factor_corr
from ._thurstone.diagonal import diagonal_portfolio
from ._thurstone.transport import blend_correlation, transport_weights

__all__ = ["ThurstonePortfolio"]


def _pow2(n: int) -> int:
    return 1 << int(np.ceil(np.log2(max(int(n), 2))))


def _normalize(w) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    s = w.sum()
    return w / s if s > 0 else np.full(len(w), 1.0 / len(w))


class ThurstonePortfolio(BaseOnlinePortfolio):
    """Long-only Thurstone-portfolio estimator.

    Parameters
    ----------
    target : {"diagonal", "equal"} or array, default "diagonal"
        Benchmark the calibration reproduces under the reference correlation. An
        array (e.g. capitalization weights) is used as-is.
    calib : {"diagonal", "market"}, default "diagonal"
        Reference correlation ``C_calib``: identity (independent) or a one-factor
        market model fitted from the covariance.
    phi : float in [0, 1], default 1.0
        Correlation scale for the tilt: 0 reproduces the benchmark, 1 uses the
        full estimated correlation.
    n_paths : int, default 16384
        Monte-Carlo seed budget (rounded up to a power of two).
    n_quad : int, default 16
        Gauss--Hermite nodes for one-factor calibration.
    seed : int, default 42
        Seed for the fixed path ensemble.
    covariance : estimator or None, default None
        Online covariance estimator (``partial_fit`` / ``covariance_``). None uses
        a built-in EWMA with ``halflife``.
    halflife : float, default 60.0
        EWMA halflife when ``covariance is None``.
    """

    def __init__(
        self,
        *,
        target="diagonal",
        calib: str = "diagonal",
        phi: float = 1.0,
        n_paths: int = 1 << 14,
        n_quad: int = 16,
        seed: int = 42,
        covariance_estimator=None,
        halflife: float = 60.0,
    ):
        super().__init__(covariance_estimator=covariance_estimator, halflife=halflife)
        self.target = target
        self.calib = calib
        self.phi = phi
        self.n_paths = n_paths
        self.n_quad = n_quad
        self.seed = seed
        # persistent state set in _cold_start
        self._seeds = None
        self._ability = None
        self._C_calib = None
        self._target_w = None
        self._betas = None

    # ------------------------------------------------------------- helpers
    def _resolve_target(self, cov: np.ndarray) -> np.ndarray:
        n = cov.shape[0]
        if isinstance(self.target, str):
            if self.target == "diagonal":
                return diagonal_portfolio(cov)
            if self.target == "equal":
                return np.full(n, 1.0 / n)
            raise ValueError(f"unknown target {self.target!r}")
        return _normalize(self.target)

    # ------------------------------------------------------- estimator hooks
    def _cold_start(self, cov: np.ndarray) -> None:
        if not 0.0 <= self.phi <= 1.0:
            raise ValueError("phi must lie in [0, 1].")
        n = cov.shape[0]
        base = base_density()
        tgt = self._resolve_target(cov)
        self._target_w = tgt

        if self.calib == "diagonal":
            self._betas = np.zeros(n)
            self._C_calib = np.eye(n)
            self._ability = calibrate_diagonal(tgt, base=base)
        elif self.calib == "market":
            b = market_betas(cov, weights=tgt)
            self._betas = b
            self._C_calib = one_factor_corr(b)
            self._ability = calibrate_one_factor(tgt, b, base=base, n_quad=self.n_quad)
        else:
            raise ValueError(f"unknown calib {self.calib!r} (use 'diagonal' or 'market')")

        rng = np.random.default_rng(self.seed)
        self._seeds = rng.standard_normal((_pow2(self.n_paths), n))
        self._online_update(cov)

    def _online_update(self, cov: np.ndarray) -> None:
        C_tilt = blend_correlation(self._C_calib, cov, self.phi)
        self._weights = transport_weights(self._ability, C_tilt, self._seeds)

    # --------------------------------------------------- fitted attributes
    @property
    def ability_(self) -> np.ndarray:
        return self._ability

    @property
    def betas_(self) -> np.ndarray:
        return self._betas

    @property
    def target_(self) -> np.ndarray:
        return self._target_w
