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
from ._thurstone.calibrate import calibrate_diagonal, calibrate_one_factor
from ._thurstone.covariance import cov_to_corr, factor_decompose, market_betas, one_factor_corr
from ._thurstone.diagonal import diagonal_portfolio
from ._thurstone.transport import (
    DEFAULT_PATHS,
    path_budget,
    blend_correlation,
    race_weights,
    transport_weights,
    transport_weights_lowrank,
    transport_weights_lowrank_t,
    transport_weights_t,
)

__all__ = ["ThurstonePortfolio"]




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
    sampler : {"gaussian", "student_t"} or callable, default "gaussian"
        Distribution of the latent race. ``"gaussian"`` is the classic Thurstone
        normal race. ``"student_t"`` runs a multivariate-t race (``nu`` degrees of
        freedom): fat marginal tails *and* tail dependence, so the winning
        probabilities become a genuine tail statistic rather than a function of
        the correlation alone -- the same common-seed transport keeps it smooth.
        A **callable** ``(ability, corr, seeds) -> X`` drives the race with *any*
        centered performance law (a copula with real tail dependence, a skew-t, a
        structured generator) -- the "any simulation" path; it must be a
        deterministic function of the fixed ``seeds`` (the common-seed contract;
        see :func:`allocation._thurstone.transport.gaussian_sampler` for the
        reference and the contract). Custom samplers run in dense mode only
        (``factors=None``).
    nu : float, default 7.0
        Degrees of freedom for ``sampler="student_t"`` (must be > 0; smaller is
        heavier-tailed, ``nu -> inf`` recovers the Gaussian race). Ignored for the
        Gaussian sampler.
    n_paths : int, default 65536
        Monte-Carlo seed budget (rounded up to a power of two).
    factors : int or None, default None
        If set, run the tilt with a ``k``-factor (low-rank) correlation and the
        ``O(M n k)`` transport instead of the dense ``O(M n^2) + O(n^3)`` one --
        what makes the tilt usable on very large universes (e.g. Russell-3000),
        where every inversion-based allocator is undefined. ``None`` uses the
        exact dense transport.
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
        sampler: str = "gaussian",
        nu: float = 7.0,
        n_paths: int = DEFAULT_PATHS,
        factors: int | None = None,
        seed: int = 42,
        covariance_estimator=None,
        halflife: float = 60.0,
    ):
        super().__init__(covariance_estimator=covariance_estimator, halflife=halflife)
        self.target = target
        self.calib = calib
        self.phi = phi
        self.sampler = sampler
        self.nu = nu
        self.n_paths = n_paths
        self.factors = factors
        self.seed = seed
        # persistent state set in _cold_start
        self._seeds = None
        self._seeds_factor = None
        self._seeds_idio = None
        self._seeds_chi2 = None
        self._custom_sampler = False
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
        self._custom_sampler = callable(self.sampler)
        if self._custom_sampler:
            if self.factors:
                raise ValueError(
                    "a callable sampler is supported only in dense mode (set factors=None)"
                )
        elif self.sampler not in ("gaussian", "student_t"):
            raise ValueError(
                f"unknown sampler {self.sampler!r} (use 'gaussian', 'student_t', or a callable)"
            )
        if self.sampler == "student_t" and not self.nu > 0:
            raise ValueError("nu must be > 0 for the student_t sampler.")
        n = cov.shape[0]
        tgt = self._resolve_target(cov)
        self._target_w = tgt

        if self.calib == "diagonal":
            self._betas = np.zeros(n)
            self._C_calib = np.eye(n)
            self._ability = calibrate_diagonal(tgt)
        elif self.calib == "market":
            b = market_betas(cov, weights=tgt)
            self._betas = b
            self._C_calib = one_factor_corr(b)
            self._ability = calibrate_one_factor(tgt, b)
        else:
            raise ValueError(f"unknown calib {self.calib!r} (use 'diagonal' or 'market')")

        m = path_budget(self.n_paths)
        rng = np.random.default_rng(self.seed)
        if self.factors:
            k = min(int(self.factors), n)
            # 2k columns, because the tilt is blended in factor space and the
            # blend of two k-factor correlations has rank 2k. See
            # _reference_factors for why the blend is not factored directly.
            self._seeds_factor = rng.standard_normal((m, 2 * k))
            self._seeds_idio = rng.standard_normal((m, n))
            self._factor_rank = k
            self._ref_factors = self._reference_factors(n, k)
        else:
            self._seeds = rng.standard_normal((m, n))
        # fixed per-path t-mixing scalar -- drawn once, like every other seed, so
        # the t-race stays smooth in the correlation (turnover tracks structure).
        if self.sampler == "student_t":
            self._seeds_chi2 = rng.chisquare(self.nu, m)
        self._online_update(cov)

    def _reference_factors(self, n, k):
        """A factor form of the calibration reference, exact where one exists.

        The independent reference is B = 0, d = 1 exactly. Handing the identity
        to a truncating factoriser instead is what issue #57 is about, so it is
        never done here. A one-factor market reference likewise has an exact
        rank-one form, taken from the betas rather than recovered numerically.
        """
        C = np.asarray(self._C_calib, dtype=float)
        if np.allclose(C, np.eye(n), atol=1e-12):
            return np.zeros((n, 0)), np.ones(n)
        if self._betas is not None:
            b = np.asarray(self._betas, dtype=float).reshape(-1, 1)
            return b, np.clip(1.0 - b[:, 0] ** 2, 1e-6, None)
        return factor_decompose(C, k, seed=self.seed)

    def _online_update(self, cov: np.ndarray) -> None:
        if self._custom_sampler:
            # the "any simulation" path: drive the race with a caller-supplied
            # centered law, scored by the universal argmin core.
            C_tilt = blend_correlation(self._C_calib, cov, self.phi)
            X = self.sampler(self._ability, C_tilt, self._seeds)
            self._weights = race_weights(X)
            return
        if self.factors:
            # Blend in factor space rather than factoring the blend.
            #
            # Factoring the blend breaks the anchor. At phi = 0 the blend is the
            # calibration reference, and for calib="diagonal" that is the
            # identity, whose eigendecomposition has no distinguished leading
            # direction: truncating it to k factors and restoring the diagonal
            # invents correlation out of an arbitrary basis choice. An equal
            # target at zero confidence came back as weights spread from 0.110
            # to 0.149 instead of 0.125 (issue #57).
            #
            # Two k-factor correlations blend exactly, and stay in the family:
            #
            #   (1-phi)(B0 B0' + diag(d0)) + phi (B1 B1' + diag(d1))
            #     = [sqrt(1-phi) B0 | sqrt(phi) B1][.]'
            #       + diag((1-phi) d0 + phi d1)
            #
            # which is rank 2k, exact at both ends and at every point between,
            # with no eigendecomposition of the blend anywhere.
            k = min(int(self.factors), cov.shape[0])
            B0, d0 = self._ref_factors
            B1, d1 = factor_decompose(cov_to_corr(cov), k, seed=self.seed)
            B = np.hstack([np.sqrt(1.0 - self.phi) * B0, np.sqrt(self.phi) * B1])
            d = (1.0 - self.phi) * d0 + self.phi * d1
            if B.shape[1] < self._seeds_factor.shape[1]:
                B = np.hstack([B, np.zeros((B.shape[0],
                                            self._seeds_factor.shape[1] - B.shape[1]))])
            if self.sampler == "student_t":
                self._weights = transport_weights_lowrank_t(
                    self._ability, B, d, self._seeds_factor, self._seeds_idio,
                    self._seeds_chi2, self.nu,
                )
            else:
                self._weights = transport_weights_lowrank(
                    self._ability, B, d, self._seeds_factor, self._seeds_idio
                )
        else:
            C_tilt = blend_correlation(self._C_calib, cov, self.phi)
            if self.sampler == "student_t":
                self._weights = transport_weights_t(
                    self._ability, C_tilt, self._seeds, self._seeds_chi2, self.nu
                )
            else:
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
