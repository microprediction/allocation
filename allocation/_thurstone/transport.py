"""Common-seed transport: realize the correlated race with fixed seeds.

The portfolio weights are the win frequencies of a correlated race among assets.
We evaluate them by Monte Carlo with a *fixed* ensemble of seeds, so that as the
correlation drifts the weights move smoothly (low turnover): see the smoothness
theorem in the accompanying paper.

The race factors cleanly into two steps:

1. **draw** a performance matrix ``X`` of shape ``(M, n)`` whose rows are
   correlated performances -- one row per Monte-Carlo path; and
2. **score** it: the argmin of each row wins, and the win frequencies are the
   weights (:func:`_race_weights`).

Step 2 is universal; step 1 is the *sampler*. Any sampler may be plugged in --
Gaussian (default), Student-t / t-copula (fat marginal tails + tail dependence),
or anything else -- subject to one hard contract: **a sampler must be a
deterministic function of a fixed seed ensemble.** All randomness enters only
through seeds drawn once (at cold start); nothing samples fresh randomness per
call. That invariant is what keeps common-seed transport smooth -- turnover then
tracks how much the *structure* moved, not sampling noise. A sampler that draws
its own randomness would pass every correctness test yet silently destroy the
turnover guarantee, so it is a contract, not a suggestion.
"""

from __future__ import annotations

import numpy as np

from .covariance import cov_to_corr, nearest_correlation

__all__ = [
    "symmetric_sqrt",
    "blend_correlation",
    "transport_weights",
    "transport_weights_lowrank",
    "transport_weights_t",
    "transport_weights_lowrank_t",
    "transport_weights_lowrank_blockt",
    "transport_weights_lowrank_custom",
    "race_weights",
    "gaussian_sampler",
]


def _race_weights(X: np.ndarray) -> np.ndarray:
    """Win frequencies of a performance matrix: the argmin of each row wins.

    ``X`` has shape ``(M, n)`` (``M`` paths, ``n`` assets); smaller performance
    wins, matching the ability convention (smaller ability == stronger). Returns
    a length-``n`` simplex vector.
    """
    winners = np.argmin(X, axis=1)
    counts = np.bincount(winners, minlength=X.shape[1]).astype(float)
    total = counts.sum()
    return counts / total if total > 0 else np.full(X.shape[1], 1.0 / X.shape[1])


def race_weights(X: np.ndarray) -> np.ndarray:
    """Public alias of the race-scoring core (see :func:`_race_weights`).

    Win frequencies of a performance matrix ``X`` of shape ``(M, n)``: the argmin
    of each row wins. Use this to build a *fully custom* race -- draw ``X`` from any
    centered law (keyed to a fixed seed ensemble) and score it here.
    """
    return _race_weights(X)


def gaussian_sampler(ability: np.ndarray, corr: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Reference race sampler: rows ``~ N(ability, corr)``, recoloured from the
    fixed standard-normal ``seeds`` (shape ``(M, n)``).

    The signature ``(ability, corr, seeds) -> X`` is the **custom-sampler
    contract** consumed by :class:`~allocation.ThurstonePortfolio` (``sampler=``).
    Any callable with this signature that is a *deterministic* function of the
    fixed ``seeds`` may be supplied to drive the race with an arbitrary centered
    performance law -- a copula with genuine tail dependence, a skew-t, or a
    structured generator. That determinism (no fresh randomness per call) is what
    preserves common-seed transport, hence low turnover; a sampler that draws its
    own randomness would pass correctness tests yet silently inflate turnover. The
    returned ``X`` is added to ``ability`` (the location) and scored by
    :func:`race_weights`.
    """
    return np.asarray(ability, dtype=float) + seeds @ symmetric_sqrt(np.asarray(corr, dtype=float))


def _t_scale(seeds_chi2: np.ndarray, nu: float) -> np.ndarray:
    """Per-path Student-t mixing scale ``sqrt(W / nu)``, ``W ~ chi^2_nu``.

    Dividing a correlated Gaussian draw by this column turns it into a
    multivariate Student-t with ``nu`` degrees of freedom: fat marginal tails
    *and* tail dependence (a small ``W`` inflates a whole path at once, so the
    assets co-move in the extreme). ``seeds_chi2`` is a fixed ``(M,)`` ensemble,
    so the t-race is as smooth in the correlation as the Gaussian one.
    """
    return np.sqrt(np.asarray(seeds_chi2, dtype=float) / float(nu))[:, None]


def symmetric_sqrt(C: np.ndarray) -> np.ndarray:
    """Symmetric (PCA) square root of a correlation matrix.

    Smoother in ``C`` than a Cholesky factor (continuous except at eigenvalue
    crossings), which is what keeps the recoloured paths -- and hence the
    weights -- smooth as ``C`` drifts.
    """
    vals, vecs = np.linalg.eigh(C)
    return (vecs * np.sqrt(np.clip(vals, 1e-12, None))) @ vecs.T


def blend_correlation(C_calib: np.ndarray, cov: np.ndarray, phi: float) -> np.ndarray:
    """Tilt correlation ``C_tilt = nearest_corr((1-phi) C_calib + phi corr(cov))``.

    ``phi = 0`` recovers the reference (and so reproduces the target); ``phi = 1``
    uses the full estimated correlation.
    """
    Ct = (1.0 - phi) * np.asarray(C_calib, dtype=float) + phi * cov_to_corr(cov)
    return nearest_correlation(Ct)


def transport_weights(ability: np.ndarray, C_tilt: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Win frequencies of ``N(ability, C_tilt)`` using the fixed ``seeds``.

    ``seeds`` has shape ``(M, n)``; the same seeds across calls is what makes the
    weights move smoothly with ``C_tilt`` (turnover tracks correlation change,
    not sampling noise). The minimum performance wins.
    """
    a = np.asarray(ability, dtype=float)
    S = symmetric_sqrt(C_tilt)
    X = a + seeds @ S  # rows ~ N(a, C_tilt)
    return _race_weights(X)


def transport_weights_t(ability, C_tilt, seeds, seeds_chi2, nu):
    """Win frequencies of a multivariate-t race ``t_nu(ability, C_tilt)``.

    Same correlated Gaussian draw as :func:`transport_weights`, divided by the
    fixed per-path t-scale -- so the marginals are Student-t and the paths share
    a common scale shock (tail dependence). As ``nu -> inf`` this converges to
    the Gaussian race. ``seeds`` is ``(M, n)`` and ``seeds_chi2`` is ``(M,)``,
    both fixed across calls.
    """
    a = np.asarray(ability, dtype=float)
    S = symmetric_sqrt(C_tilt)
    X = a + (seeds @ S) / _t_scale(seeds_chi2, nu)  # rows ~ t_nu(a, C_tilt)
    return _race_weights(X)


def transport_weights_lowrank(ability, loadings, idio, seeds_factor, seeds_idio):
    """Win frequencies of ``N(ability, diag(idio) + B B^T)`` in ``O(M n k)``.

    For a ``k``-factor tilt correlation ``C = diag(idio) + B B^T`` (``B`` is
    ``loadings``, shape ``(n, k)``), a draw is ``x = a + B z + sqrt(idio) * eps``
    with ``z ~ N(0, I_k)``, ``eps ~ N(0, I_n)`` -- no ``n x n`` square root, so the
    race costs ``O(M n k)`` instead of the dense ``O(M n^2) + O(n^3)``. This is
    what lets the Thurstone tilt scale to thousands of names (Russell-3000). The
    *same* fixed ``seeds_factor`` / ``seeds_idio`` across calls keeps weights
    smooth, exactly as in the dense transport.
    """
    a = np.asarray(ability, dtype=float)
    B = np.asarray(loadings, dtype=float)
    s = np.sqrt(np.clip(np.asarray(idio, dtype=float), 0.0, None))
    X = a + seeds_factor @ B.T + seeds_idio * s  # (M, n), rows ~ N(a, diag(idio) + B B^T)
    return _race_weights(X)


def transport_weights_lowrank_t(ability, loadings, idio, seeds_factor, seeds_idio, seeds_chi2, nu):
    """Multivariate-t low-rank race: ``t_nu(ability, diag(idio) + B B^T)`` in ``O(M n k)``.

    The factor analogue of :func:`transport_weights_t` -- the same ``O(M n k)``
    Gaussian draw as :func:`transport_weights_lowrank`, divided by the fixed
    per-path t-scale. Keeps the heavy-tailed tilt usable on large universes (no
    ``n x n`` square root) while injecting fat tails and tail dependence.
    """
    a = np.asarray(ability, dtype=float)
    B = np.asarray(loadings, dtype=float)
    s = np.sqrt(np.clip(np.asarray(idio, dtype=float), 0.0, None))
    X = a + (seeds_factor @ B.T + seeds_idio * s) / _t_scale(seeds_chi2, nu)
    return _race_weights(X)


def transport_weights_lowrank_blockt(
    ability, loadings, idio, seeds_factor, seeds_idio, seeds_chi2, nus
):
    """Factor race where **each factor carries its own tail index** ``nus[f]``.

    Factors are standardized to unit variance, so the covariance contribution is
    ``B B^T + diag(idio)`` for *any* ``nus`` -- correlation is held fixed while
    the per-factor tail is swept. A heavy factor (``2 < nus[f] < inf``) induces
    lower-tail dependence among the assets that load on it *without touching the
    covariance*: precisely the knob the tail-consistency theorem needs. The tail
    is **block-specific** -- an asset that does not load on a heavy factor stays
    tail-independent of that block. ``nus[f] = inf`` is a Gaussian factor, and
    with all ``nus = inf`` this reduces exactly to
    :func:`transport_weights_lowrank`.

    Shapes: ``loadings (n, k)``, ``seeds_factor (M, k)``, ``seeds_idio (M, n)``,
    ``seeds_chi2 (M, k)`` (one fixed chi-square stream per factor; columns for
    Gaussian factors are ignored), ``nus`` length ``k`` (each ``> 2`` or ``inf``).
    """
    a = np.asarray(ability, dtype=float)
    B = np.asarray(loadings, dtype=float)
    s = np.sqrt(np.clip(np.asarray(idio, dtype=float), 0.0, None))
    nus = np.asarray(nus, dtype=float)
    F = np.asarray(seeds_factor, dtype=float).copy()  # (M, k); standard normal
    finite = np.isfinite(nus)
    if np.any(finite):
        w = np.asarray(seeds_chi2, dtype=float)[:, finite]  # (M, kf) ~ chi^2_nu
        nf = nus[finite]
        # z / sqrt(W/nu) is Student-t_nu (variance nu/(nu-2)); rescale to unit
        # variance so the factor's covariance contribution is tail-invariant.
        F[:, finite] = (F[:, finite] / np.sqrt(w / nf)) * np.sqrt((nf - 2.0) / nf)
    X = a + F @ B.T + seeds_idio * s
    return _race_weights(X)


def transport_weights_lowrank_custom(ability, loadings, idio, factor, seeds_idio):
    """Win frequencies of a factor race with a **caller-supplied factor matrix**.

    The most general factor race: the caller builds the ``(M, k)`` standardized
    factor draws ``factor`` however it likes -- Gaussian, Student-t, skew-t, a
    copula, a structured permutation generator, anything -- and this just colours
    them through the loadings, adds Gaussian idiosyncratic noise, and scores the
    argmin. As long as ``factor`` is a fixed ensemble, common-seed transport
    still holds. (For the covariance to equal ``B B^T + diag(idio)`` the factor
    columns should be standardized to zero mean / unit variance; whether to do so
    is the caller's modelling choice.)

    Shapes: ``loadings (n, k)``, ``factor (M, k)``, ``seeds_idio (M, n)``.
    """
    a = np.asarray(ability, dtype=float)
    B = np.asarray(loadings, dtype=float)
    s = np.sqrt(np.clip(np.asarray(idio, dtype=float), 0.0, None))
    X = a + np.asarray(factor, dtype=float) @ B.T + seeds_idio * s
    return _race_weights(X)
