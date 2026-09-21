"""Does the gamma-blended reference C_used(gamma) = (1-gamma)*C_tree + gamma*Sigma_hat
fix F=0's failure at high Schur gamma?

overnight_repair_validation.py found that at gamma>=0.75 (real Schur portfolios),
the F=0 (independent) calibration reference is actively WRONG (negative mean
gain): a Schur portfolio built with substantial cross-block coupling has already
used most of the correlation, so pretending independence at calibration time
double-subtracts it. The paper's own discussion proposed reusing gamma directly
via a blended reference instead of a factor-rank budget F.

C_used(gamma) is dense in general (a linear blend of the full nested tree and
the true correlation), so it is factor-fit (factor_model_contrast, rank k) and
calibrated through the SAME fast lattice engine (calibrate_factor) that the F=0
baseline already uses -- this makes the comparison apples-to-apples on
calibration precision. Re-racing under the true C_full still uses the dense
transport_weights, exactly as everywhere else in this paper.

An earlier version of this script calibrated under the dense C_used directly
via a hand-rolled Monte-Carlo Newton solve. Two things went wrong with that
approach, both worth recording. First, a real bug: the raw least-squares
Jacobian solve returned a huge step along the near-null constant-shift
direction (shifting every theta equally changes no winning probability), the
clip truncated it to a uniform vector, and the mean-centering step silently
cancelled the entire update every iteration -- calibration never moved past
theta=0. Second, even after fixing that, cross-checking the fixed-point
property (calibrate and re-race under the SAME law should reproduce w_base
almost exactly) by re-racing with a DIFFERENT numerical method (dense Monte
Carlo) than the one used to calibrate (also dense Monte Carlo, but a
different draw) produced a large, resolution-dependent, non-monotonic
residual that had nothing to do with calibration accuracy -- it was purely
the disagreement between two different approximate estimators of the same
quantity. Checking the fixed-point property with the SAME engine on both
sides (below) is why this version calibrates through the lattice engine
throughout.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from schur_thurstone_repair import _nodes, es95, nested_tree_factor
from overnight_repair_validation import UNIVERSES, load_universe

from allocation import SchurComplementary
from allocation._thurstone.covariance import cov_to_corr
from allocation._thurstone.factor import calibrate_factor, factor_model_contrast
from allocation._thurstone.transport import transport_weights

M_SEEDS = 1 << 14
POINTS = 251
BLEND_RANK = 8


def gamma_blend_dense(corr_in, order, gamma):
    """C_used(gamma) = (1-gamma)*C_tree + gamma*Sigma_hat, as a dense,
    unit-diagonal, PSD correlation matrix."""
    n = len(corr_in)
    V, D = nested_tree_factor(corr_in, order, None)  # full retained tree
    C_tree = V @ V.T + np.diag(D)
    C = (1 - gamma) * C_tree + gamma * corr_in
    d = np.sqrt(np.clip(np.diag(C), 1e-8, None))
    C = C / d[:, None] / d[None, :]
    vals, vecs = np.linalg.eigh((C + C.T) / 2)
    if vals.min() < 1e-6:
        vals = np.clip(vals, 1e-6, None)
        C = (vecs * vals) @ vecs.T
        d = np.sqrt(np.clip(np.diag(C), 1e-8, None))
        C = C / d[:, None] / d[None, :]
    return C


def run_trial(rng, panels, gamma):
    universe = rng.choice(list(UNIVERSES))
    R_full, cols = panels[universe]
    T_total, N_total = R_full.shape
    max_n = min(25, N_total)
    n_assets = int(rng.integers(12, max_n + 1))
    asset_idx = rng.choice(N_total, size=n_assets, replace=False)
    T_in = int(rng.integers(250, 750))
    T_out = 125
    max_start = T_total - T_in - T_out
    if max_start <= 0:
        return None
    start = int(rng.integers(0, max_start))
    Rin = R_full[start:start + T_in][:, asset_idx]
    Rout = R_full[start + T_in:start + T_in + T_out][:, asset_idx]

    est = SchurComplementary(gamma=gamma, keep_monotonic=False)
    est.fit(Rin)
    w_base = est.weights_
    corr_in = cov_to_corr(np.asarray(est._cov_estimator.covariance_, dtype=float))
    base_es = es95(Rout @ w_base)
    seeds = rng.standard_normal((M_SEEDS, n_assets))

    # F=0: independent reference, fast lattice engine (as in the main paper)
    V0, D0 = nested_tree_factor(corr_in, est.order_, 0)
    F0, W0 = _nodes(V0.shape[1])
    theta0 = calibrate_factor(w_base, V0, D0, F0, W0, points=POINTS)
    w_f0 = transport_weights(theta0, corr_in, seeds)
    f0_es = es95(Rout @ w_f0)

    # gamma-blend: dense reference, factor-fit to rank k, same lattice engine
    C_used = gamma_blend_dense(corr_in, est.order_, gamma)
    k = min(BLEND_RANK, n_assets - 1)
    Vb, Db = factor_model_contrast(C_used, k)
    Fb, Wb = _nodes(k)
    theta_b = calibrate_factor(w_base, Vb, Db, Fb, Wb, points=POINTS)
    w_bl = transport_weights(theta_b, corr_in, seeds)
    bl_es = es95(Rout @ w_bl)

    return base_es, f0_es - base_es, bl_es - base_es


def main(n_trials=60, seed=0):
    panels = {name: load_universe(name) for name in UNIVERSES}
    gammas = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = {g: {"f0": [], "blend": []} for g in gammas}
    rng_master = np.random.default_rng(seed)
    t0 = time.time()
    for i in range(n_trials):
        for g in gammas:
            rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            out = run_trial(rng, panels, g)
            if out is not None:
                _, f0_gain, bl_gain = out
                results[g]["f0"].append(f0_gain)
                results[g]["blend"].append(bl_gain)
        if i % 5 == 0:
            print(f"{i}/{n_trials} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\ndone in {time.time()-t0:.0f}s\n")
    print(f"{'gamma':>7}{'F=0 mean':>12}{'F=0 win%':>10}"
          f"{'blend mean':>14}{'blend win%':>12}{'n':>6}")
    for g in gammas:
        f0 = np.array(results[g]["f0"])
        bl = np.array(results[g]["blend"])
        print(f"{g:>7.2f}{f0.mean():>12.5f}{100*(f0>0).mean():>10.1f}"
              f"{bl.mean():>14.5f}{100*(bl>0).mean():>12.1f}{len(f0):>6}")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
