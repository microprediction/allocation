"""Calibration: back out Thurstone abilities that reproduce target weights.

Two engines, selected by the structure of the reference correlation ``C_calib``:

* **diagonal** (independent field) -- the exact lattice inverse from
  :mod:`thurstone`. Cheap; this is flavour (i).
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
from thurstone import Density, Race

from .ability import base_density, state_price_implied_ability

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
    ability, betas, *, base: Density | None = None, n_quad: int = 16
) -> np.ndarray:
    """Winning probabilities under a one-factor race, by quadrature.

    Model: ``X_i = a_i + b_i Z + sqrt(1 - b_i^2) eps_i`` with ``Z ~ N(0,1)`` the
    common factor and ``eps_i`` independent. Conditional on ``Z = z`` the field
    is independent, so the exact lattice race applies; we integrate over ``z``
    with Gauss--Hermite (probabilists') quadrature.
    """
    base = base if base is not None else base_density()
    lat = base.lattice
    a = np.asarray(ability, dtype=float)
    b = np.clip(np.asarray(betas, dtype=float), -0.999, 0.999)
    s = np.sqrt(np.clip(1.0 - b ** 2, 1e-6, 1.0))

    nodes, qw = np.polynomial.hermite_e.hermegauss(n_quad)
    qw = qw / np.sqrt(2.0 * np.pi)  # so weights sum to 1

    n = len(a)
    acc = np.zeros(n, dtype=float)
    for z, w in zip(nodes, qw):
        densities = [
            Density.skew_normal(lat, loc=float(a[i] + b[i] * z), scale=float(s[i]), a=0.0)
            for i in range(n)
        ]
        acc += w * np.asarray(Race(densities).state_prices(), dtype=float)
    return _normalize(np.clip(acc, 0.0, None))


def calibrate_diagonal(target, *, base: Density | None = None, n_iter: int = 4) -> np.ndarray:
    """Abilities reproducing ``target`` under an independent field (flavour i).

    Exact lattice inverse via the :mod:`thurstone` calibrator.
    """
    return state_price_implied_ability(_normalize(target), base=base, n_iter=n_iter)


def calibrate_one_factor(
    target,
    betas,
    *,
    base: Density | None = None,
    n_quad: int = 16,
    n_iter: int = 60,
    step: float = 0.5,
    tol: float = 1e-4,
) -> np.ndarray:
    """Abilities reproducing ``target`` under a one-factor race (flavour ii).

    Damped fixed-point on the quadrature forward map. Because winning
    probability is monotone *decreasing* in ability (min wins), we nudge
    ``a_i`` up when the model over-prices asset ``i`` and down when it
    under-prices it, on a log scale, re-centering each step (abilities are only
    identified up to a constant).
    """
    target = _normalize(target)
    base = base if base is not None else base_density()
    log_t = np.log(np.clip(target, 1e-12, None))

    a = calibrate_diagonal(target, base=base)  # warm start (independent inverse)
    for _ in range(n_iter):
        p = winprobs_one_factor(a, betas, base=base, n_quad=n_quad)
        if np.max(np.abs(p - target)) < tol:
            break
        log_p = np.log(np.clip(p, 1e-12, None))
        a = a + step * (log_p - log_t)  # p decreasing in a -> this is a descent step
        a = a - np.median(a)
    return a
