"""k-factor Thurstone races: forward, calibration, and derivatives.

Thin adapter over the canonical implementation in the `winning` package
(`winning.factor.core`), which is where this engine is developed and fixed.
Kept under allocation's own names (`winprobs_factor`, `calibrate_factor`, ...)
so callers here are unaffected by upstream renames.

Model: X_i = a_i + v_i . f + sqrt(D_i) eps_i, f ~ N(0, I_k). Convention:
minimum performance wins; smaller ability = stronger (following the
kinetics factor-probit work, Cotton 2026, "Scalable Probit Calibration";
algorithms originate in the `winning`/`thurstone` lattice transform, SIAM
J. Financial Mathematics 2021).

Why the old preconditioner worked (the observation recorded in
experiments/preconditioner.py that the choice-space Jacobian has condition
number near 1 even when cond(C) ~ 800): the Jacobian is minus a weighted
graph Laplacian whose edge weight between assets i and j is the
photo-finish density (the probability density of a tie for the win). Its
conditioning is governed by how evenly photo-finish mass spreads across
pairs, not by the spectrum of C.

Riding on `winning` directly (rather than a local duplicate) also picks up
its N=2 fix: the generic damped-Jacobi calibration loop two-cycles on a
2-name universe (K_2 is bipartite; the normalized photo-finish Laplacian
eigenvalue is exactly 2), so `winning` special-cases it with the closed
form instead of silently returning a wrong answer at the iteration cap.
"""

from __future__ import annotations

import numpy as np
from winning.factor.core import (
    abilities_from_probabilities_factor,
    factor_model_contrast as _factor_model_contrast,
    hermite_nodes as _hermite_nodes,
    jacobian_vector_product,
    win_probabilities_factor,
)

__all__ = [
    "winprobs_factor",
    "calibrate_factor",
    "jacobian_vector_product",
    "factor_model_contrast",
    "hermite_nodes",
]

hermite_nodes = _hermite_nodes
factor_model_contrast = _factor_model_contrast


def winprobs_factor(ability, V, D, F, W, points: int = 1501) -> np.ndarray:
    """Normalized min-wins winning probabilities under the k-factor model."""
    return win_probabilities_factor(ability, V, D, F, W, points=points)


def calibrate_factor(target, V, D, F, W, n_iter: int = 50, tol: float = 1e-6,
                     return_info: bool = False, points: int = 1501):
    """Abilities reproducing `target` win probabilities (Newton on the
    factor race, N=2 closed form, independent-inverse warm start)."""
    return abilities_from_probabilities_factor(
        target, V, D, F, W, n_iter=n_iter, tol=tol,
        return_info=return_info, points=points)
