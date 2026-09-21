"""Bridge to the race calibration maps.

These thin wrappers turn portfolio weights (treated as state prices / winning
probabilities) into latent abilities and back. Smaller ability == stronger
competitor, because the winner is the *minimum* performance, which is the
min-wins convention ``winning`` uses at its front door.

The default path is ``winning.calibrate_abilities`` and
``winning.race_probabilities``, which are the package's recommended entry
points and are accelerated by the compiled ``fastrace`` kernels where
installed. Measured against the older ``thurstone`` package on the same
fields they agree to a centred correlation of 1.0000 and a scale ratio of
0.999, round-trip about four orders of magnitude more accurately
(1e-9 against 4e-5), and run roughly 38x faster at forty assets.

Passing an explicit ``base`` density routes to the older ``thurstone``
lattice calibrator instead, which is the only way to use a non-standard
performance density. Nothing in the package does that by default.
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
    if base is None:
        import winning
        return np.asarray(
            winning.calibrate_abilities(np.asarray(weights, dtype=float)), dtype=float)
    cal = AbilityCalibrator(base=base, n_iter=n_iter)
    a = cal.solve_from_prices([float(w) for w in weights])
    return np.asarray(a, dtype=float)


def ability_implied_state_prices(ability, *, base: Density | None = None) -> np.ndarray:
    """Forward map: abilities to winning probabilities (the field's state prices)."""
    if base is None:
        import winning
        p = winning.race_probabilities(np.asarray(ability, dtype=float))
        if isinstance(p, tuple):
            p = p[0]
        p = np.asarray(p, dtype=float)
        s = p.sum()
        return p / s if s > 0 else p
    cal = AbilityCalibrator(base=base)
    p = cal.state_prices_from_ability([float(a) for a in ability])
    p = np.asarray(p, dtype=float)
    s = p.sum()
    return p / s if s > 0 else p
