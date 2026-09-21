"""Does minimum-variance ever actually LOSE to equal-weight/HRP out of
sample -- and if so, is that exactly where the repair helps it?

Every check so far asked "does the repair help minimum-variance," never
"does minimum-variance's own optimization actually beat the naive
baselines it's supposed to beat, in the regime tested." The classic result
(DeMiguel, Garlappi & Uppal 2009, "Optimal Versus Naive Diversification,"
already in this paper's refs.bib) is that 1/N beats mean-variance-style
optimization once the covariance is noisy enough -- but that needs a
genuinely extreme observations-per-asset ratio, and the earlier checks here
used T_in/n around 6:1 to 60:1, never pushed toward 1:1 or below.

This sweeps T_in/n from generous to genuinely extreme (T_in as low as 30
against n up to 40) on real data, reporting, at each point:
  (1) MinimumVariance (unshrunk) vs EqualWeight, head to head, no repair
      at all -- does the classic "naive beats optimized" flip ever show up;
  (2) the SAME independent-reference repair applied to MinimumVariance,
      at that same noise level -- does it help more (or only) exactly
      where minimum-variance was already losing.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from schur_thurstone_repair import _nodes, es95
from overnight_repair_validation import UNIVERSES, load_universe

from allocation import BoxConstrained, EqualWeight, MinimumVariance
from allocation._thurstone.covariance import cov_to_corr
from allocation._thurstone.factor import calibrate_factor
from allocation._thurstone.transport import transport_weights

M_SEEDS = 1 << 13
POINTS = 251


def independent_repair(w_base, corr_target, seeds):
    n = len(w_base)
    V = np.zeros((n, 1))
    D = np.ones(n)
    Fq, Wq = _nodes(1)
    theta = calibrate_factor(w_base, V, D, Fq, Wq, points=POINTS)
    return transport_weights(theta, corr_target, seeds)


def run_trial(rng, panels, n_assets, T_in, universe="sp500"):
    R_full, cols = panels[universe]
    T_total, N_total = R_full.shape
    if n_assets > N_total:
        return None
    asset_idx = rng.choice(N_total, size=n_assets, replace=False)
    T_out = 125
    max_start = T_total - T_in - T_out
    if max_start <= 0:
        return None
    start = int(rng.integers(0, max_start))
    Rin = R_full[start:start + T_in][:, asset_idx]
    Rout = R_full[start + T_in:start + T_in + T_out][:, asset_idx]

    eq = EqualWeight(); eq.fit(Rin)
    w_eq = eq.weights_
    es_eq = es95(Rout @ w_eq)

    mv = BoxConstrained(MinimumVariance(shrinkage=0.0))
    try:
        mv.fit(Rin)
    except Exception:
        return None
    w_mv = np.clip(mv.weights_, 0.0, None)
    if w_mv.sum() <= 0:
        return None
    w_mv = w_mv / w_mv.sum()
    es_mv = es95(Rout @ w_mv)

    corr_in = cov_to_corr(np.asarray(mv.estimator._cov_estimator.covariance_, dtype=float))
    seeds = rng.standard_normal((M_SEEDS, n_assets))
    try:
        w_rep = independent_repair(w_mv, corr_in, seeds)
    except Exception:
        return None
    es_rep = es95(Rout @ w_rep)

    return es_mv - es_eq, es_rep - es_mv  # (mv vs eq raw, repair effect on mv)


def main(n_trials=150, seed=0):
    panels = {name: load_universe(name) for name in UNIVERSES}
    grid = [
        (15, 750, "mixed"), (15, 250, "mixed"), (15, 90, "mixed"), (15, 40, "mixed"),
        (30, 750, "mixed"), (30, 250, "mixed"), (30, 90, "mixed"), (30, 40, "mixed"),
        (100, 750, "sp500"), (100, 250, "sp500"), (100, 90, "sp500"),
        (200, 750, "sp500"), (200, 250, "sp500"), (200, 90, "sp500"),
        (400, 750, "sp500"), (400, 250, "sp500"),
    ]
    print(f"{'n':>4}{'T_in':>6}{'T/n':>7}"
          f"{'mv-vs-eq mean':>15}{'mv-vs-eq win%':>15}"
          f"{'repair mean':>13}{'repair win%':>13}{'trials':>8}", flush=True)
    t0 = time.time()
    for n_assets, T_in, uni in grid:
        mv_vs_eq, rep_eff = [], []
        rng_master = np.random.default_rng(seed)
        for _ in range(n_trials):
            rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            universe = rng.choice(list(UNIVERSES)) if uni == "mixed" else uni
            out = run_trial(rng, panels, n_assets, T_in, universe=universe)
            if out is not None:
                a, b = out
                mv_vs_eq.append(a)
                rep_eff.append(b)
        mv_vs_eq = np.array(mv_vs_eq)
        rep_eff = np.array(rep_eff)
        ratio = T_in / n_assets
        print(f"{n_assets:>4}{T_in:>6}{ratio:>7.1f}"
              f"{mv_vs_eq.mean():>15.5f}{100*(mv_vs_eq>0).mean():>15.1f}"
              f"{rep_eff.mean():>13.5f}{100*(rep_eff>0).mean():>13.1f}"
              f"{len(mv_vs_eq):>8}", flush=True)
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
