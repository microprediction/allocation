"""Bridge to the :mod:`thurstone` calibration maps.

These thin wrappers turn portfolio weights (treated as state prices / winning
probabilities) into latent Thurstone abilities and back, using the lattice
machinery in the ``thurstone`` package. Smaller ability == stronger competitor,
because the winner is the *minimum* performance.
"""

from __future__ import annotations

import numpy as np
from thurstone import AbilityCalibrator, Density, UniformLattice
from thurstone.conventions import STD_L, STD_SCALE, STD_UNIT

__all__ = [
    "base_density",
    "state_price_implied_ability",
    "ability_implied_state_prices",
]


def base_density(L: int = STD_L, unit: float = STD_UNIT, scale: float = STD_SCALE) -> Density:
    """Standard symmetric performance density on the default lattice."""
    lat = UniformLattice(L, unit)
    return Density.skew_normal(lat, loc=0.0, scale=scale, a=0.0)


def state_price_implied_ability(
    weights, *, base: Density | None = None, n_iter: int = 4
) -> np.ndarray:
    """Invert weights (as winning probabilities) to abilities, up to a constant."""
    base = base if base is not None else base_density()
    cal = AbilityCalibrator(base=base, n_iter=n_iter)
    a = cal.solve_from_prices([float(w) for w in weights])
    return np.asarray(a, dtype=float)


def ability_implied_state_prices(ability, *, base: Density | None = None) -> np.ndarray:
    """Forward map: abilities to winning probabilities (the field's state prices)."""
    base = base if base is not None else base_density()
    cal = AbilityCalibrator(base=base)
    p = cal.state_prices_from_ability([float(a) for a in ability])
    p = np.asarray(p, dtype=float)
    s = p.sum()
    return p / s if s > 0 else p
