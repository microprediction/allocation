"""River-style streaming portfolio over a *changing* universe.

Real backtests cannot assume a fixed asset set: index members are added and
dropped, names list and delist. Following river, the streaming interface is
dict-based -- ``learn_one(x)`` with ``x = {asset_id: return}`` and
``predict_one()`` returning ``{asset_id: weight}`` -- so a changing universe is
handled natively: it is just a changing set of dict keys.

Per-asset state is keyed by id and carried across updates. The common-seed
transport composes with this: each asset owns a persistent seed column, so a
continuing name keeps its column (and a smooth weight), a new name gets a fresh
column, and a dropped name's column simply falls out of the active set.
"""

from __future__ import annotations

import zlib

import numpy as np

from .baselines import risk_parity_weights
from .convex import (
    max_decorrelation_weights,
    max_diversification_weights,
    mean_variance_weights,
    min_variance_weights,
)
from ._schur.coupling import compute_monotonic_weights, compute_weights
from ._schur.seriation import seriate
from ._thurstone.ability import base_density
from ._thurstone.calibrate import calibrate_diagonal, calibrate_one_factor
from ._thurstone.covariance import market_betas, one_factor_corr
from ._thurstone.diagonal import diagonal_portfolio
from ._thurstone.transport import blend_correlation, transport_weights

__all__ = [
    "KeyedEwmaCovariance",
    "StreamingThurstone",
    "StreamingSchur",
    "StreamingHRP",
    "StreamingEqualWeight",
    "StreamingInverseVariance",
    "StreamingRiskParity",
    "StreamingMinimumVariance",
    "StreamingMaximumDiversification",
    "StreamingMaximumDecorrelation",
    "StreamingMeanVariance",
]


def _halflife_to_alpha(halflife: float) -> float:
    return 1.0 - 0.5 ** (1.0 / max(float(halflife), 1e-9))


class KeyedEwmaCovariance:
    """Dict-keyed exponentially-weighted online covariance.

    ``learn_one({id: value})`` updates means and pairwise covariances keyed by
    asset id; ``matrix(ids)`` returns the covariance submatrix for any set of
    ids. Assets may appear or disappear between calls.
    """

    def __init__(self, halflife: float = 60.0):
        self.halflife = halflife
        self.mean: dict = {}
        self.cov: dict = {}  # frozenset-like sorted (i, j) -> ewma covariance
        self.count: dict = {}

    def learn_one(self, x: dict) -> "KeyedEwmaCovariance":
        a = _halflife_to_alpha(self.halflife)
        for k, v in x.items():
            self.mean[k] = v if k not in self.mean else (1 - a) * self.mean[k] + a * v
            self.count[k] = self.count.get(k, 0) + 1
        dev = {k: x[k] - self.mean[k] for k in x}
        keys = list(x.keys())
        for i in keys:
            di = dev[i]
            for j in keys:
                if j < i:
                    continue
                key = (i, j)
                self.cov[key] = (1 - a) * self.cov.get(key, 0.0) + a * di * dev[j]
        return self

    def matrix(self, ids) -> np.ndarray:
        ids = list(ids)
        n = len(ids)
        C = np.zeros((n, n), dtype=float)
        for a, i in enumerate(ids):
            for b in range(a, n):
                j = ids[b]
                key = (i, j) if i <= j else (j, i)
                val = self.cov.get(key, 0.0)
                C[a, b] = C[b, a] = val
        # guard a strictly-positive diagonal for assets seen at least once
        d = np.diag(C).copy()
        d[d <= 0] = 1e-8
        np.fill_diagonal(C, d)
        return C


class StreamingThurstone:
    """Streaming Thurstone portfolio over a changing universe (river-style).

    Parameters mirror :class:`allocation.ThurstonePortfolio`. The active universe
    at each step is the set of ids present in the latest observation.
    """

    def __init__(
        self,
        *,
        calib: str = "diagonal",
        phi: float = 1.0,
        n_paths: int = 1 << 12,
        n_quad: int = 16,
        halflife: float = 60.0,
        seed: int = 42,
        min_obs: int = 20,
    ):
        if not 0.0 <= phi <= 1.0:
            raise ValueError("phi must lie in [0, 1].")
        self.calib = calib
        self.phi = phi
        self.n_paths = 1 << int(np.ceil(np.log2(max(int(n_paths), 2))))
        self.n_quad = n_quad
        self.halflife = halflife
        self.seed = seed
        self.min_obs = min_obs
        self._cov = KeyedEwmaCovariance(halflife=halflife)
        self._seed_bank: dict = {}
        self._weights: dict = {}
        self._base = base_density()
        self._n = 0

    # --------------------------------------------------------- seed bank
    def _seed_column(self, asset_id) -> np.ndarray:
        col = self._seed_bank.get(asset_id)
        if col is None:
            # stable, per-asset, reproducible across sessions
            h = zlib.crc32(str(asset_id).encode()) & 0xFFFFFFFF
            rng = np.random.default_rng((self.seed, h))
            col = rng.standard_normal(self.n_paths)
            self._seed_bank[asset_id] = col
        return col

    def _seeds(self, ids) -> np.ndarray:
        return np.column_stack([self._seed_column(k) for k in ids])

    # --------------------------------------------------------- streaming
    def learn_one(self, x: dict) -> "StreamingThurstone":
        self._cov.learn_one(x)
        self._n += 1
        # only allocate to assets that have warmed up (a reliable variance);
        # this is what keeps a freshly-listed name from hijacking the
        # inverse-variance target with a cold, near-zero variance estimate.
        warm = [k for k in sorted(x.keys()) if self._cov.count.get(k, 0) >= self.min_obs]
        if len(warm) >= 2:
            self._recompute(warm)
        return self

    def _recompute(self, ids) -> None:
        cov = self._cov.matrix(ids)
        tgt = diagonal_portfolio(cov)
        if self.calib == "market":
            betas = market_betas(cov, weights=tgt)
            C_calib = one_factor_corr(betas)
            ability = calibrate_one_factor(tgt, betas, base=self._base, n_quad=self.n_quad)
        else:
            C_calib = np.eye(len(ids))
            ability = calibrate_diagonal(tgt, base=self._base)
        C_tilt = blend_correlation(C_calib, cov, self.phi)
        w = transport_weights(ability, C_tilt, self._seeds(ids))
        self._weights = dict(zip(ids, w))

    def predict_one(self, x: dict | None = None) -> dict:
        """Current portfolio as ``{asset_id: weight}`` (a copy)."""
        return dict(self._weights)

    @property
    def weights(self) -> dict:
        return dict(self._weights)


class StreamingSchur:
    """Streaming Schur-complementary portfolio over a changing universe.

    River-style ``learn_one({asset: ret})`` / ``predict_one()``. The Fiedler
    seriation is carried as a per-asset dict so its sign (hence the order) stays
    stable as the universe changes. Parameters mirror
    :class:`allocation.SchurComplementary`.
    """

    def __init__(
        self,
        *,
        gamma: float = 0.5,
        keep_monotonic: bool = True,
        knn: int | None = None,
        halflife: float = 60.0,
        min_obs: int = 20,
    ):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1].")
        self.gamma = gamma
        self.keep_monotonic = keep_monotonic
        self.knn = knn
        self.halflife = halflife
        self.min_obs = min_obs
        self._cov = KeyedEwmaCovariance(halflife=halflife)
        self._fiedler: dict = {}  # asset_id -> Fiedler coordinate (for sign stability)
        self._weights: dict = {}
        self._n = 0

    def learn_one(self, x: dict) -> "StreamingSchur":
        self._cov.learn_one(x)
        self._n += 1
        warm = [k for k in sorted(x.keys()) if self._cov.count.get(k, 0) >= self.min_obs]
        if len(warm) >= 2:
            self._recompute(warm)
        return self

    def _recompute(self, ids) -> None:
        cov = self._cov.matrix(ids)
        prev = (
            np.array([self._fiedler.get(k, 0.0) for k in ids])
            if self._fiedler
            else None
        )
        order, v = seriate(cov, previous=prev, knn=self.knn)
        self._fiedler = dict(zip(ids, v))
        if self.keep_monotonic:
            w, _ = compute_monotonic_weights(order, cov, self.gamma)
        else:
            w = compute_weights(order, cov, self.gamma)
        self._weights = dict(zip(ids, w))

    def predict_one(self, x: dict | None = None) -> dict:
        return dict(self._weights)

    @property
    def weights(self) -> dict:
        return dict(self._weights)


class StreamingHRP(StreamingSchur):
    """Streaming dynamic HRP over a changing universe.

    The ``gamma=0`` special case of :class:`StreamingSchur`: recursive-bisection
    risk parity (HRP) over the smooth Fiedler order, so the dendrogram churn of
    classic HRP is replaced by low-turnover updates. Named for recognisability.
    """

    def __init__(self, *, knn: int | None = None, halflife: float = 60.0, min_obs: int = 20):
        super().__init__(
            gamma=0.0, keep_monotonic=False, knn=knn, halflife=halflife, min_obs=min_obs
        )


class StreamingEqualWeight:
    """Streaming equal-weight portfolio over a changing universe.

    Uniform over the names that have warmed up (seen ``>= min_obs`` times), so a
    freshly-listed name does not jump straight to full weight. Needs no
    covariance; only per-asset counts.
    """

    def __init__(self, *, min_obs: int = 20):
        self.min_obs = min_obs
        self._count: dict = {}
        self._weights: dict = {}

    def learn_one(self, x: dict) -> "StreamingEqualWeight":
        for k in x:
            self._count[k] = self._count.get(k, 0) + 1
        warm = [k for k in sorted(x) if self._count.get(k, 0) >= self.min_obs]
        if warm:
            w = 1.0 / len(warm)
            self._weights = {k: w for k in warm}
        return self

    def predict_one(self, x: dict | None = None) -> dict:
        return dict(self._weights)

    @property
    def weights(self) -> dict:
        return dict(self._weights)


class StreamingInverseVariance:
    """Streaming inverse-variance (naive risk-parity) over a changing universe."""

    def __init__(self, *, halflife: float = 60.0, min_obs: int = 20):
        self.halflife = halflife
        self.min_obs = min_obs
        self._cov = KeyedEwmaCovariance(halflife=halflife)
        self._weights: dict = {}

    def learn_one(self, x: dict) -> "StreamingInverseVariance":
        self._cov.learn_one(x)
        warm = [k for k in sorted(x) if self._cov.count.get(k, 0) >= self.min_obs]
        if len(warm) >= 1:
            var = np.array([max(self._cov.cov.get((k, k), 0.0), 1e-12) for k in warm])
            inv = 1.0 / var
            w = inv / inv.sum()
            self._weights = dict(zip(warm, w))
        return self

    def predict_one(self, x: dict | None = None) -> dict:
        return dict(self._weights)

    @property
    def weights(self) -> dict:
        return dict(self._weights)


class StreamingRiskParity:
    """Streaming equal-risk-contribution (ERC) portfolio over a changing universe.

    The unnormalised ERC iterate is carried per asset id, so continuing names
    warm-start from their previous coordinate (smooth, low turnover) while a new
    name starts from an inverse-volatility default. Parameters mirror
    :class:`allocation.RiskParity`.
    """

    def __init__(self, *, budgets=None, halflife: float = 60.0, min_obs: int = 20):
        self.budgets = budgets
        self.halflife = halflife
        self.min_obs = min_obs
        self._cov = KeyedEwmaCovariance(halflife=halflife)
        self._x: dict = {}  # asset_id -> last unnormalised ERC coordinate (warm start)
        self._weights: dict = {}

    def learn_one(self, x: dict) -> "StreamingRiskParity":
        self._cov.learn_one(x)
        warm = [k for k in sorted(x) if self._cov.count.get(k, 0) >= self.min_obs]
        if len(warm) >= 2:
            self._recompute(warm)
        return self

    def _recompute(self, ids) -> None:
        cov = self._cov.matrix(ids)
        x0 = np.array([self._x.get(k, np.nan) for k in ids])  # NaN -> inv-vol default
        w, xs = risk_parity_weights(cov, budgets=self.budgets, x0=x0)
        self._x = dict(zip(ids, xs))
        self._weights = dict(zip(ids, w))

    def predict_one(self, x: dict | None = None) -> dict:
        return dict(self._weights)

    @property
    def weights(self) -> dict:
        return dict(self._weights)


class _StreamingClosedForm:
    """Shared plumbing for streaming closed-form allocators over a changing universe.

    Subclasses implement ``_weights_from_cov(cov)`` returning weights aligned to
    the warm id order. Weights may be signed (these are unconstrained).
    """

    def __init__(self, *, shrinkage: float = 0.0, halflife: float = 60.0, min_obs: int = 20):
        self.shrinkage = shrinkage
        self.halflife = halflife
        self.min_obs = min_obs
        self._cov = KeyedEwmaCovariance(halflife=halflife)
        self._weights: dict = {}

    def _weights_from_cov(self, cov: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def learn_one(self, x: dict):
        self._cov.learn_one(x)
        warm = [k for k in sorted(x) if self._cov.count.get(k, 0) >= self.min_obs]
        if len(warm) >= 2:
            w = self._weights_from_cov(self._cov.matrix(warm))
            self._weights = dict(zip(warm, w))
        return self

    def predict_one(self, x: dict | None = None) -> dict:
        return dict(self._weights)

    @property
    def weights(self) -> dict:
        return dict(self._weights)


class StreamingMinimumVariance(_StreamingClosedForm):
    """Streaming unconstrained minimum-variance over a changing universe (signed)."""

    def _weights_from_cov(self, cov: np.ndarray) -> np.ndarray:
        return min_variance_weights(cov, self.shrinkage)


class StreamingMaximumDiversification(_StreamingClosedForm):
    """Streaming unconstrained maximum-diversification over a changing universe (signed)."""

    def _weights_from_cov(self, cov: np.ndarray) -> np.ndarray:
        return max_diversification_weights(cov, self.shrinkage)


class StreamingMaximumDecorrelation(_StreamingClosedForm):
    """Streaming maximum-decorrelation over a changing universe (signed)."""

    def _weights_from_cov(self, cov: np.ndarray) -> np.ndarray:
        return max_decorrelation_weights(cov, self.shrinkage)


class StreamingMeanVariance:
    """Streaming tangency / mean-variance portfolio over a changing universe (signed).

    Uses the keyed EWMA mean for ``mu`` unless ``expected_returns`` (a
    ``{id: mu}`` dict) is given. Mirrors :class:`allocation.MeanVariance`.
    """

    def __init__(
        self, *, expected_returns=None, shrinkage: float = 0.0, halflife: float = 60.0, min_obs: int = 20
    ):
        self.expected_returns = expected_returns
        self.shrinkage = shrinkage
        self.halflife = halflife
        self.min_obs = min_obs
        self._cov = KeyedEwmaCovariance(halflife=halflife)
        self._weights: dict = {}

    def learn_one(self, x: dict) -> "StreamingMeanVariance":
        self._cov.learn_one(x)
        warm = [k for k in sorted(x) if self._cov.count.get(k, 0) >= self.min_obs]
        if len(warm) >= 2:
            cov = self._cov.matrix(warm)
            if self.expected_returns is not None:
                mu = np.array([self.expected_returns.get(k, 0.0) for k in warm])
            else:
                mu = np.array([self._cov.mean.get(k, 0.0) for k in warm])
            w = mean_variance_weights(cov, mu, self.shrinkage)
            self._weights = dict(zip(warm, w))
        return self

    def predict_one(self, x: dict | None = None) -> dict:
        return dict(self._weights)

    @property
    def weights(self) -> dict:
        return dict(self._weights)
