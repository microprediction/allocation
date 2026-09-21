"""Calibration: back out Thurstone abilities that reproduce target weights.

Two engines, selected by the structure of the reference correlation ``C_calib``:

* **diagonal** (independent field) -- the exact lattice inverse from
  ``winning``. Cheap; this is flavour (i).
* **one-factor** -- a single common factor with per-asset loadings ``betas``.
  Conditional on the factor the assets are independent, so the race is evaluated
  by Gauss--Hermite quadrature over the factor (``winprobs_one_factor``); the
  inverse is a damped fixed-point on that forward map. This is flavour (ii), and
  the quadrature is exactly the calibration tool.

Convention throughout: the *minimum* performance wins, so a **smaller** ability
means a **stronger** competitor (higher winning probability).
"""

from __future__ import annotations

import numpy as np
import winning

from .ability import state_price_implied_ability

__all__ = [
    "winprobs_one_factor",
    "calibrate_diagonal",
    "calibrate_one_factor",
]


def _normalize(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    s = w.sum()
    return w / s if s > 0 else np.full(len(w), 1.0 / len(w))


def winprobs_one_factor(
    ability, betas
) -> np.ndarray:
    """Winning probabilities under a one-factor race, by quadrature.

    Model: ``X_i = a_i + b_i Z + sqrt(1 - b_i^2) eps_i`` with ``Z ~ N(0,1)`` the
    common factor and ``eps_i`` independent. Conditional on ``Z = z`` the field
    is independent, so the exact lattice race applies; the race is evaluated by
    ``winning`` with the loadings as ``V`` and the idiosyncratic variances as
    ``D``.
    """
    a = np.asarray(ability, dtype=float)
    b = np.clip(np.asarray(betas, dtype=float), -0.999, 0.999)
    # V is a column of loadings and D the idiosyncratic variances; passing V
    # alone leaves D at its default and inflates the total variance, which is
    # a silent 6e-2 error against the model this function documents.
    p = winning.race_probabilities(a, V=b.reshape(-1, 1), D=1.0 - b ** 2)
    if isinstance(p, tuple):
        p = p[0]
    return _normalize(np.clip(np.asarray(p, dtype=float), 0.0, None))


def calibrate_diagonal(target) -> np.ndarray:
    """Abilities reproducing ``target`` under an independent field (flavour i).

    Exact inverse via ``winning.calibrate_abilities``.
    """
    return state_price_implied_ability(_normalize(target))


def calibrate_one_factor(target, betas) -> np.ndarray:
    """Abilities reproducing ``target`` under a one-factor race (flavour ii).

    Solved directly by ``winning.calibrate_abilities`` with the factor loading
    passed as ``V``, replacing a damped fixed point on a quadrature forward map
    that left about three percent of error at its tolerance. Abilities are only
    identified up to a constant, so the result is re-centred.
    """
    target = _normalize(target)
    b = np.clip(np.asarray(betas, dtype=float), -0.999, 0.999)
    # the same floor the independent flavour applies; winning raises on a
    # zero target by design, and a benchmark with one zero-weight name is
    # ordinary, so both flavours must agree rather than differ by a string
    a = np.asarray(
        winning.calibrate_abilities(
            np.maximum(target, 1e-12), V=b.reshape(-1, 1), D=1.0 - b ** 2,
            target_floor=1e-12),
        dtype=float)
    return a - np.median(a)
