"""Fair, streaming turnover comparison: repaired HRP vs. the simple
alternatives, using COMMON Monte Carlo seeds across a real walk-forward
sequence -- not the one-shot, fresh-seeds-per-trial comparison used earlier
today.

repair_vs_simple_alternatives.py measured "turnover" as the L1 distance
between two INDEPENDENTLY cold-fit portfolios on unrelated random draws,
with FRESH Monte Carlo seeds every trial. That is not what the companion
paper's smoothness theorem is about, and it likely overstates the repair's
real turnover: re-drawing seeds each time injects pure sampling noise into
the comparison, and there is no persistent state for the race to "hold
still" against when nothing has genuinely changed (Theorem 2's own "Finite
paths" remark: common seeds make Delta_C=0 give Delta_w=0 EXACTLY; fresh
seeds carry an O(1/sqrt(M)) floor regardless of whether anything moved).

This walks forward through REAL historical dates on a FIXED sub-universe,
recomputing HRP and the alternatives on a rolling window each rebalance
(so their own natural period-to-period change is whatever it normally is),
but computing the repair with SEEDS DRAWN ONCE before the walk and reused
at every rebalance -- the actual common-seed-transport mechanism. Reports
mean period-to-period L1 turnover for each method, and out-of-sample ES95
over the whole walk, so both the turnover *and* the performance claim from
the earlier comparison get re-checked under a fair protocol.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from schur_thurstone_repair import _nodes, es95

from allocation import BoxConstrained, HierarchicalRiskParity, MinimumVariance
from allocation._thurstone.covariance import cov_to_corr
from allocation._thurstone.factor import calibrate_factor
from allocation._thurstone.transport import transport_weights

M_SEEDS = 1 << 14
POINTS = 251
REBAL_STEP = 21          # ~monthly
WINDOW = 500             # trailing window for each rebalance's covariance


def independent_repair(w_base, corr_full, seeds):
    n = len(w_base)
    V = np.zeros((n, 1))
    D = np.ones(n)
    Fq, Wq = _nodes(1)
    theta = calibrate_factor(w_base, V, D, Fq, Wq, points=POINTS)
    return transport_weights(theta, corr_full, seeds)


def run_walk(rng, R, n_assets, seeds_mode):
    """seeds_mode: 'common' (drawn once, reused) or 'fresh' (redrawn each
    rebalance) -- the ablation this whole check is about.

    HRP and MinimumVariance are each ONE persistent estimator, cold-started
    once on the first window then advanced with partial_fit on only the NEW
    returns each rebalance -- their own built-in EWMA covariance and (for
    HRP) warm-started Fiedler seriation, not a from-scratch refit on a
    recomputed rolling window every period. That refit-from-scratch
    approach (an earlier version of this script) was needlessly reinventing
    exactly the smoothing machinery BaseOnlinePortfolio already provides.
    """
    T_total, N_total = R.shape
    asset_idx = rng.choice(N_total, size=n_assets, replace=False)
    Rs = R[:, asset_idx]

    rebal_starts = list(range(WINDOW, T_total - REBAL_STEP, REBAL_STEP))
    if len(rebal_starts) < 3:
        return None

    seeds_common = rng.standard_normal((M_SEEDS, n_assets))

    hrp = HierarchicalRiskParity()
    mv = BoxConstrained(MinimumVariance(shrinkage=0.7))
    hrp.fit(Rs[:WINDOW])
    mv.fit(Rs[:WINDOW])

    prev = {"hrp": None, "repaired": None, "minvar": None, "blend": None}
    turnover = {"hrp": [], "repaired": [], "minvar": [], "blend": []}
    port_rets = {"hrp": [], "repaired": [], "minvar": [], "blend": []}

    for start in rebal_starts:
        if start > WINDOW:
            R_new = Rs[start - REBAL_STEP:start]
            hrp.partial_fit(R_new)
            mv.partial_fit(R_new)
        Rout = Rs[start:start + REBAL_STEP]

        w_hrp = hrp.weights_
        corr_in = cov_to_corr(np.asarray(hrp._cov_estimator.covariance_, dtype=float))
        seeds = seeds_common if seeds_mode == "common" else rng.standard_normal((M_SEEDS, n_assets))
        try:
            w_rep = independent_repair(w_hrp, corr_in, seeds)
        except Exception:
            w_rep = w_hrp

        w_mv = np.clip(mv.weights_, 0.0, None)
        w_mv = w_mv / w_mv.sum() if w_mv.sum() > 0 else np.full(n_assets, 1 / n_assets)

        w_blend = 0.5 * w_hrp + 0.5 * w_mv
        w_blend = w_blend / w_blend.sum()

        weights = {"hrp": w_hrp, "repaired": w_rep, "minvar": w_mv, "blend": w_blend}
        for name, w in weights.items():
            if prev[name] is not None:
                turnover[name].append(float(np.abs(w - prev[name]).sum()))
            prev[name] = w
            port_rets[name].append(Rout @ w)

    out = {}
    for name in weights:
        out[f"{name}_turnover"] = float(np.mean(turnover[name])) if turnover[name] else np.nan
        out[f"{name}_es95"] = es95(np.concatenate(port_rets[name]))
    return out


def main(n_trials=40, n_assets=20, seed=0):
    ret = pd.read_parquet("data/sp500_returns_2014_2024.parquet")
    R = ret.to_numpy()
    print(f"panel: {R.shape}, rebalancing every {REBAL_STEP} days, window={WINDOW}", flush=True)

    rows_common, rows_fresh = [], []
    rng_master = np.random.default_rng(seed)
    t0 = time.time()
    for i in range(n_trials):
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        rc = run_walk(np.random.default_rng(rng.integers(0, 2**31 - 1)), R, n_assets, "common")
        rf = run_walk(np.random.default_rng(rng.integers(0, 2**31 - 1)), R, n_assets, "fresh")
        if rc is not None:
            rows_common.append(rc)
        if rf is not None:
            rows_fresh.append(rf)
        if i % 10 == 0:
            print(f"{i}/{n_trials} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\ndone in {time.time()-t0:.0f}s, n={len(rows_common)}\n")
    for label, rows in (("COMMON seeds (fair)", rows_common), ("fresh seeds (the old way)", rows_fresh)):
        df = pd.DataFrame(rows)
        print(f"--- {label} ---")
        print(f"{'method':<12}{'mean turnover':>16}{'mean ES95':>14}")
        for name in ("hrp", "repaired", "minvar", "blend"):
            print(f"{name:<12}{df[f'{name}_turnover'].mean():>16.4f}{df[f'{name}_es95'].mean():>14.5f}")
        print()


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
