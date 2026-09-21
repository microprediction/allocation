"""Bridge to the race calibration maps in :mod:`winning`.

These thin wrappers turn portfolio weights (treated as state prices / winning
probabilities) into latent abilities and back. Smaller ability means a stronger
competitor, because the winner is the *minimum* performance, which is the
min-wins convention ``winning`` uses at its front door.

Both directions call ``winning``'s recommended entry points,
:func:`winning.calibrate_abilities` and :func:`winning.race_probabilities`,
which are accelerated by the compiled ``fastrace`` kernels where installed.
They are neither ``winning.classic``, whose primitive is empirical atoms
rather than formulas, nor ``winning.research``, the density-agnostic engine
that the old ``thurstone`` package aliased.
"""

from __future__ import annotations

import numpy as np
import winning

__all__ = [
    "base_density",
    "state_price_implied_ability",
    "ability_implied_state_prices",
]


def base_density(*_args, **_kwargs):
    """Retained for compatibility. ``winning`` carries its own base law, so
    there is no density object to construct; callers pass nothing."""
    return None


def _as_probabilities(p) -> np.ndarray:
    if isinstance(p, tuple):
        p = p[0]
    p = np.asarray(p, dtype=float)
    s = p.sum()
    return p / s if s > 0 else np.full(len(p), 1.0 / len(p))


def state_price_implied_ability(
    weights, *, base=None, n_iter: int = 4, floor: float = 1e-12
) -> np.ndarray:
    """Invert weights (as winning probabilities) to abilities, up to a constant.

    A zero weight has no finite inverse, and ``winning`` raises on one rather
    than inventing a value. Portfolio weights with exact zeros are ordinary, so
    they are floored first; the result is then a one-sided bound on the floored
    contrasts. The lattice calibrator this replaced returned a grid-edge
    constant instead, which was junk that never announced itself.
    """
    w = np.asarray(weights, dtype=float)
    if (w <= 0).any():
        w = np.maximum(w, floor)
    return np.asarray(winning.calibrate_abilities(w), dtype=float)


def ability_implied_state_prices(ability, *, base=None) -> np.ndarray:
    """Forward map: abilities to winning probabilities (the field's state prices)."""
    return _as_probabilities(winning.race_probabilities(np.asarray(ability, dtype=float)))
