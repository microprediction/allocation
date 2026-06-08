"""Trading-cost control: a quadratic turnover penalty around any allocator.

A quadratic transaction cost ``lam * ||w - w_prev||^2`` (the market-impact model:
impact grows with trade size, so cost grows with its square) added to a local
quadratic fit ``||w - w_target||^2`` of the base objective has the closed-form
minimiser

    w = (w_target + lam * w_prev) / (1 + lam) = alpha * w_target + (1 - alpha) * w_prev,

with ``alpha = 1 / (1 + lam)``. So the cost term simply blends the base
allocator's fresh target with the previously held weights. It is smooth (linear
in both), it preserves the budget (a convex combination of two vectors that each
sum to one again sums to one, signed weights included), and it scales the step
exactly: ``||w_t - w_prev|| = alpha * ||w_target - w_prev||``.

This composes with the package's *implicit* turnover control (common-seed
transport, Fiedler-sign stability): those make the target itself smooth; this
adds an explicit, tunable damping on top, and wraps any estimator unchanged.
"""

from __future__ import annotations

import numpy as np

from .base import BaseOnlinePortfolio

__all__ = ["TurnoverPenalty", "StreamingTurnoverPenalty"]


def _alpha(cost: float) -> float:
    if cost < 0:
        raise ValueError("cost (lambda) must be non-negative.")
    return 1.0 / (1.0 + float(cost))


class TurnoverPenalty(BaseOnlinePortfolio):
    """Wrap a batch estimator with a quadratic turnover penalty.

    Parameters
    ----------
    estimator : BaseOnlinePortfolio
        The base allocator. Its weights are the per-step target; this wrapper
        damps the move toward them. Used directly (not cloned).
    cost : float >= 0, default 1.0
        Trading-cost weight ``lambda``. ``0`` leaves the base unchanged; larger
        values trade less (``alpha = 1 / (1 + cost)`` of each step is taken).
    """

    def __init__(self, estimator: BaseOnlinePortfolio, *, cost: float = 1.0):
        super().__init__()
        self.estimator = estimator
        self.cost = cost

    @property
    def alpha_(self) -> float:
        """Fraction of each target step actually taken, ``1 / (1 + cost)``."""
        return _alpha(self.cost)

    def fit(self, X, y=None):
        self.estimator.fit(X)
        self.n_features_in_ = getattr(self.estimator, "n_features_in_", None)
        self._weights = np.asarray(self.estimator.weights_, dtype=float).copy()
        return self

    def partial_fit(self, X, y=None):
        if self._weights is None:
            return self.fit(X)
        prev = self._weights
        self.estimator.partial_fit(X)
        target = np.asarray(self.estimator.weights_, dtype=float)
        a = self.alpha_
        self._weights = a * target + (1.0 - a) * prev
        return self


class StreamingTurnoverPenalty:
    """Wrap a streaming (river-style) estimator with a quadratic turnover penalty.

    Mirrors :class:`TurnoverPenalty` for the dict interface over a changing
    universe. A continuing name is blended toward its previous weight; a new name
    is taken at the target (nothing to damp toward); a departed name falls out.
    The blend is renormalised to sum to one because the active set can change.
    """

    def __init__(self, estimator, *, cost: float = 1.0):
        self.estimator = estimator
        self.cost = cost
        self._weights: dict = {}

    @property
    def alpha_(self) -> float:
        return _alpha(self.cost)

    def learn_one(self, x: dict) -> "StreamingTurnoverPenalty":
        prev = self._weights
        self.estimator.learn_one(x)
        target = self.estimator.predict_one()
        if not target:
            return self
        a = self.alpha_
        blended = {
            k: a * v + (1.0 - a) * prev.get(k, v)  # new name: prev defaults to target
            for k, v in target.items()
        }
        s = sum(blended.values())
        if s != 0:
            blended = {k: v / s for k, v in blended.items()}  # active set may have changed
        self._weights = blended
        return self

    def predict_one(self, x: dict | None = None) -> dict:
        return dict(self._weights)

    @property
    def weights(self) -> dict:
        return dict(self._weights)
