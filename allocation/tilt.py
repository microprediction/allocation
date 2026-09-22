"""The tilting round trip: weights and abilities are one object in two charts.

A long-only portfolio on the simplex interior and a vector of latent abilities
carry the same information. :func:`abilities_from_weights` and
:func:`weights_from_abilities` are inverse to each other, so a user can check
the claim in one line:

    >>> import numpy as np
    >>> from allocation import abilities_from_weights, weights_from_abilities
    >>> w = np.array([0.4, 0.25, 0.2, 0.15])
    >>> float(np.abs(weights_from_abilities(abilities_from_weights(w)) - w).max()) < 1e-8
    True

:func:`tilt_weights` is the object the round trip is for. It calibrates
abilities so a race under a reference correlation reproduces the benchmark you
hand it, then re-runs that race under the estimated correlation, damped by
``phi``. At ``phi = 0`` it recovers the benchmark and at ``phi = 1`` it races
under the full estimate. Nothing is inverted at any point, so the whole dial is
available where a covariance matrix is singular.

The race is simulated, so with the Gaussian sampler ``phi = 0`` recovers the
benchmark only to about ``1/sqrt(n_paths)``, not exactly. Paths are cheap, so
raise ``n_paths`` if the effect you are measuring is small: the default of
65536 puts the recovery error near 0.002 and quadrupling the budget halves it.

With ``sampler='student_t'`` there is an additional offset of a percent or two
that does **not** shrink with the path budget. The abilities are calibrated so
a *Gaussian* race reproduces the benchmark, and a t race at the same abilities
is a different law rather than a noisy version of the same one, so it lands
somewhere else by construction. Both races are normalised to unit variance, so
this is a shape difference and not a scale error. Read the t results as
relative to the t race at ``phi = 0``, not to the benchmark.

Two conventions worth knowing, because both are easy to trip over.

Abilities are identified only up to an additive constant, and the race is
shift-invariant. These functions return the mean-zero representative. Compare
abilities within a call, not across calls, unless you re-centre yourself.

Smaller ability means a *stronger* competitor, because the winner is the
minimum performance. That is the min-wins convention of the underlying
``winning`` package.

A zero weight has no finite inverse. :func:`abilities_from_weights` floors the
input rather than raising, so the result for a zero-weight name is a one-sided
bound and not its inverse. Pass ``floor`` to control it.
"""

from __future__ import annotations

import numpy as np

from ._thurstone.ability import (
    ability_implied_state_prices as _weights_from_abilities,
    state_price_implied_ability as _abilities_from_weights,
)
import winning

from ._thurstone.transport import (
    DEFAULT_PATHS,
    blend_correlation,
    path_budget,
    transport_weights,
    transport_weights_t,
)

__all__ = [
    "abilities_from_weights",
    "weights_from_abilities",
    "tilt_weights",
    "blend_correlation",
]


def abilities_from_weights(weights, *, floor: float = 1e-12) -> np.ndarray:
    """Abilities whose race reproduces ``weights``, mean-zero, smaller is stronger.

    Inverse of :func:`weights_from_abilities`. Weights are normalised to sum to
    one first. Non-positive entries are floored at ``floor``, since a zero share
    has no finite inverse; the corresponding ability is then a one-sided bound.
    """
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    if s > 0:
        w = w / s
    a = _abilities_from_weights(w, floor=floor)
    return a - a.mean()


def weights_from_abilities(ability) -> np.ndarray:
    """Winning probabilities of the race at ``ability``, summing to one.

    Inverse of :func:`abilities_from_weights`. Shift-invariant, so adding a
    constant to ``ability`` leaves the result unchanged.
    """
    return _weights_from_abilities(np.asarray(ability, dtype=float))


def tilt_weights(
    weights,
    cov,
    *,
    phi: float = 1.0,
    sampler: str = "gaussian",
    nu: float = 7.0,
    n_paths: int = DEFAULT_PATHS,
    seed: int = 42,
    factors: int = 3,
) -> np.ndarray:
    """Tilt a benchmark portfolio by racing it under an estimated correlation.

    ``weights`` is any long-only benchmark, for instance equal weight, inverse
    variance or a hierarchical portfolio. ``cov`` is an estimated covariance.
    ``phi`` says how far to trust it: ``0`` recovers ``weights`` and ``1``
    races under the full estimated correlation. Recovery at ``phi = 0`` is up
    to the race's Monte Carlo error, about ``1/sqrt(n_paths)``, and for
    ``sampler='student_t'`` there is a further offset that does not shrink
    with paths, because the reference law and the tilt law differ.

    The race is driven by a fixed seed ensemble, so the result moves smoothly in
    ``cov`` and successive calls at nearby covariances give nearby portfolios.
    No matrix is inverted, so ``cov`` may be singular.

    ``sampler='exact'`` evaluates the race by quadrature instead of simulating
    it, which is what the default should probably be: it reproduces the
    benchmark at ``phi = 0`` to about ``1e-10`` rather than to
    ``1/sqrt(n_paths)``, uses no seed ensemble and no memory, and at five
    thousand assets is a hundred times faster than the path budget the
    simulation needs to be usable at all. It approximates the estimated
    correlation by ``factors`` factors, which is exact when the market has that
    many and is the only approximation in the path.

    ``sampler='student_t'`` runs a multivariate-t race with ``nu`` degrees of
    freedom, giving fat marginal tails and tail dependence rather than a
    function of the correlation alone. There is no quadrature equivalent, so
    that one is simulated.
    """
    if not 0.0 <= phi <= 1.0:
        raise ValueError("phi must lie in [0, 1].")
    if sampler == "exact":
        return _tilt_exact(weights, cov, phi, factors=factors)
    if sampler not in ("gaussian", "student_t"):
        raise ValueError(f"unknown sampler {sampler!r} (use 'gaussian' or 'student_t')")
    if sampler == "student_t" and not nu > 2:
        raise ValueError("nu must exceed 2 for the t race to have finite variance")
    w = np.asarray(weights, dtype=float)
    if not np.isfinite(w).all():
        raise ValueError("weights must be finite")
    if (w < 0).any():
        raise ValueError("weights must be non-negative; a short has no winning probability")
    if not w.sum() > 0:
        raise ValueError("weights must not be all zero")
    n = len(w)
    ability = abilities_from_weights(w)
    C_tilt = blend_correlation(np.eye(n), np.asarray(cov, dtype=float), float(phi))

    m = path_budget(n_paths)
    rng = np.random.default_rng(seed)
    seeds = rng.standard_normal((m, n))
    if sampler == "gaussian":
        return transport_weights(ability, C_tilt, seeds)
    return transport_weights_t(ability, C_tilt, seeds, rng.chisquare(nu, m), nu)


def _factor_form(cov, k):
    """A ``k``-factor correlation ``VV' + diag(D)`` with unit diagonal."""
    C = np.asarray(cov, dtype=float)
    d = np.sqrt(np.clip(np.diag(C), 1e-300, None))
    R = C / np.outer(d, d)
    ev, U = np.linalg.eigh((R + R.T) / 2)
    idx = np.argsort(ev)[::-1][:k]
    V = U[:, idx] * np.sqrt(np.clip(ev[idx], 0.0, None))
    D = np.clip(1.0 - (V ** 2).sum(1), 1e-6, None)
    scale = np.sqrt((V ** 2).sum(1) + D)
    return V / scale[:, None], D / scale ** 2


def _tilt_exact(weights, cov, phi, *, factors=3):
    """The tilt by quadrature rather than simulation.

    Blending a reference identity toward a factor correlation keeps the result
    inside the factor family exactly:

        (1-phi) I + phi (VV' + diag(D))
            = (sqrt(phi) V)(sqrt(phi) V)' + diag((1-phi) + phi D)

    so every point of the dial is an exact race with no approximation beyond
    the factor count, and ``phi = 0`` is the identity, which returns the
    benchmark.
    """
    w = np.asarray(weights, dtype=float)
    if not np.isfinite(w).all():
        raise ValueError("weights must be finite")
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    if not w.sum() > 0:
        raise ValueError("weights must not be all zero")
    w = w / w.sum()

    ability = abilities_from_weights(w)
    if phi == 0.0:
        return _normalise(winning.race_probabilities(ability))
    V, D = _factor_form(cov, min(int(factors), len(w) - 1))
    return _normalise(winning.race_probabilities(
        ability, V=np.sqrt(phi) * V, D=(1.0 - phi) + phi * D))


def _normalise(p):
    if isinstance(p, tuple):
        p = p[0]
    p = np.clip(np.asarray(p, dtype=float), 0.0, None)
    s = p.sum()
    if not s > 0:
        raise ValueError("the race returned an all-zero field")
    return p / s
