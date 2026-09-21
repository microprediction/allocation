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

import warnings

import numpy as np
import winning

__all__ = [
    "state_price_implied_ability",
    "ability_implied_state_prices",
]



def _as_probabilities(p) -> np.ndarray:
    """Normalise a forward-map result, refusing to launder a poisoned one.

    The previous fallback returned equal weight when the sum was not positive.
    Its reachable trigger was NaN rather than a zero sum, because ``nan > 0``
    is False, so a broken race came back as a clean-looking equal-weight book:
    the same junk-that-never-announces-itself this module criticises the old
    lattice calibrator for.
    """
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        raise ValueError("empty field")
    if not np.isfinite(p).all():
        raise ValueError("the race returned non-finite probabilities")
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if not s > 0:
        raise ValueError("the race returned an all-zero field")
    return p / s


def state_price_implied_ability(
    weights, *, floor: float = 1e-12
) -> np.ndarray:
    """Invert weights (as winning probabilities) to abilities, up to a constant.

    A zero weight has no finite inverse, and ``winning`` raises on one rather
    than inventing a value. Portfolio weights with exact zeros are ordinary, so
    they are floored first; the result is then a one-sided bound on the floored
    contrasts. The lattice calibrator this replaced returned a grid-edge
    constant instead, which was junk that never announced itself.
    """
    w = np.asarray(weights, dtype=float)
    if not np.isfinite(w).all():
        raise ValueError("weights must be finite")
    if (w < 0).any():
        raise ValueError("weights must be non-negative; a short has no winning probability")
    s = w.sum()
    if not s > 0:
        raise ValueError("weights must not be all zero")
    w = np.maximum(w / s, floor)
    w = w / w.sum()

    a, info = winning.calibrate_abilities(w, target_floor=floor, return_info=True)
    a = np.asarray(a, dtype=float)
    if info.get("converged", True):
        return a
    a, err = _damped_refine(a, w)
    if err > 1e-6:
        warnings.warn(
            f"ability inversion did not converge: the race at the returned "
            f"abilities reproduces the target to {err:.2e}, not to machine "
            f"precision. This happens on small, highly concentrated fields. "
            f"Treat the result as approximate.",
            RuntimeWarning, stacklevel=2)
    return a


def _damped_refine(a, target, *, n_iter: int = 400, step: float = 0.5,
                   tol: float = 1e-10):
    """Finish an inverse that ``winning`` left unconverged.

    Its undamped iteration enters a limit cycle on small fields: at three
    assets roughly half of Dirichlet draws stall, worst case round-tripping to
    an error of 0.46 with the two largest names transposed, and neither more
    iterations, a tighter tolerance nor finer quadrature moves the residual.

    Winning probability is monotone decreasing in ability, since the winner is
    the minimum performance, so nudging ability up where the race over-prices a
    name is a descent step. Damping it breaks the cycle. Abilities are
    identified up to a constant, so the iterate is re-centred each step.

    Returns ``(abilities, achieved_error)`` so the caller can tell the user
    when the answer is approximate rather than exact.
    """
    log_t = np.log(target)
    best, best_err = a, np.inf
    for _ in range(n_iter):
        p = np.asarray(winning.race_probabilities(a), dtype=float)
        err = float(np.abs(p - target).max())
        if err < best_err:
            best, best_err = a, err
        elif step > 1e-3:
            # the step overshot; halve it and resume from the best iterate,
            # which is what concentrated targets need
            step *= 0.5
            a = best
            continue
        if err < tol:
            break
        a = a + step * (np.log(np.clip(p, 1e-300, None)) - log_t)
        a = a - a.mean()
    return best, best_err


def ability_implied_state_prices(ability) -> np.ndarray:
    """Forward map: abilities to winning probabilities (the field's state prices)."""
    return _as_probabilities(winning.race_probabilities(np.asarray(ability, dtype=float)))
