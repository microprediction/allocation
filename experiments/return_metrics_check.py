"""Does the F=0 repair's tail-risk improvement (ES95) come at a cost to
mean return / Sharpe over the rest of the distribution, or is it free?

Same real-market setup as overnight_repair_validation.py, F=0 (gauss target)
only, but logging mean return, volatility, annualized Sharpe, and ES95 for
base vs. repair side by side -- not just the tail metric.
"""
from __future__ import annotations
import csv, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")

from schur_thurstone_repair import _nodes, es95, nested_tree_factor
from allocation import HierarchicalRiskParity, SchurComplementary
from allocation._thurstone.covariance import cov_to_corr
from allocation._thurstone.factor import calibrate_factor
from allocation._thurstone.transport import transport_weights
from overnight_repair_validation import UNIVERSES, load_universe

M_SEEDS = 1 << 14
POINTS = 251
OUT_CSV = "data/return_metrics_results.csv"
FIELDS = ["trial","universe","n_assets","method","mean_base","mean_rep",
          "vol_base","vol_rep","sharpe_base","sharpe_rep","es95_base","es95_rep"]

def sharpe(returns):
    sd = returns.std()
    return float(returns.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0

def run_trial(trial_id, rng, panels):
    universe = rng.choice(list(UNIVERSES))
    R_full, cols = panels[universe]
    T_total, N_total = R_full.shape
    max_n = min(40, N_total) if universe == "sp500" else min(25, N_total)
    n_assets = int(rng.integers(12, max_n + 1))
    asset_idx = rng.choice(N_total, size=n_assets, replace=False)
    T_in = int(rng.integers(250, 750)); T_out = 125
    max_start = T_total - T_in - T_out
    if max_start <= 0: return []
    start = int(rng.integers(0, max_start))
    Rin = R_full[start:start+T_in][:, asset_idx]
    Rout = R_full[start+T_in:start+T_in+T_out][:, asset_idx]
    seeds = rng.standard_normal((M_SEEDS, n_assets))
    rows = []
    for name, est in (("hrp", HierarchicalRiskParity()),
                      ("schur", SchurComplementary(gamma=0.5, keep_monotonic=False))):
        est.fit(Rin)
        w_base = est.weights_
        corr_in = cov_to_corr(np.asarray(est._cov_estimator.covariance_, dtype=float))
        V, D = nested_tree_factor(corr_in, est.order_, 0)
        Fq, Wq = _nodes(V.shape[1])
        theta = calibrate_factor(w_base, V, D, Fq, Wq, points=POINTS)
        w_rep = transport_weights(theta, corr_in, seeds)
        r_base = Rout @ w_base; r_rep = Rout @ w_rep
        rows.append(dict(trial=trial_id, universe=universe, n_assets=n_assets, method=name,
                          mean_base=r_base.mean(), mean_rep=r_rep.mean(),
                          vol_base=r_base.std(), vol_rep=r_rep.std(),
                          sharpe_base=sharpe(r_base), sharpe_rep=sharpe(r_rep),
                          es95_base=es95(r_base), es95_rep=es95(r_rep)))
    return rows

def main(n_trials=1500, seed=None):
    panels = {name: load_universe(name) for name in UNIVERSES}
    rng = np.random.default_rng(seed)
    write_header = not os.path.exists(OUT_CSV)
    t0 = time.time()
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header: w.writeheader()
        for i in range(n_trials):
            for row in run_trial(i, rng, panels):
                w.writerow(row)
            f.flush()
            if i % 100 == 0:
                print(f"{i}/{n_trials} ({time.time()-t0:.0f}s)", flush=True)
    print(f"done {n_trials} in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
