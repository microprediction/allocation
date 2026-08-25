"""What happens if you run the repair map twice (or more)?

The repair R = W_C_full o W_C_used^-1 is emphatically NOT the identity when
C_used != C_full -- that's the entire point -- so there is no a priori reason
R(R(w)) should equal R(w). This checks what iterating it actually does:
does a second application help further, do nothing (a fast approach to a
fixed point), or hurt (overshoot)? Tracks both the out-of-sample ES95 at
each iterate and the L1 weight-change per step, to see whether the map is
contracting toward a fixed point or not.

F=0 (independent calibration reference) only, since that's the headline
real-market finding; real S&P 500 / REIT sub-portfolios, same panels as
overnight_repair_validation.py. Results appended to
data/iterate_repair_results.csv.
"""
from __future__ import annotations

import csv
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from schur_thurstone_repair import _nodes, es95, nested_tree_factor

from allocation import HierarchicalRiskParity
from allocation._thurstone.covariance import cov_to_corr
from allocation._thurstone.factor import calibrate_factor
from allocation._thurstone.transport import transport_weights
from overnight_repair_validation import UNIVERSES, load_universe

N_ITER = 8
M_SEEDS = 1 << 14
POINTS = 251
OUT_CSV = "data/iterate_repair_results.csv"
FIELDS = ["trial", "universe", "n_assets", "T_in", "iter", "es95", "l1_step", "note"]


def run_trial(trial_id: int, rng: np.random.Generator, panels: dict) -> list[dict]:
    universe = rng.choice(list(UNIVERSES))
    R_full, cols = panels[universe]
    T_total, N_total = R_full.shape
    max_n = min(40, N_total) if universe == "sp500" else min(25, N_total)
    n_assets = int(rng.integers(12, max_n + 1))
    asset_idx = rng.choice(N_total, size=n_assets, replace=False)

    T_in = int(rng.integers(250, 750))
    T_out = 125
    max_start = T_total - T_in - T_out
    if max_start <= 0:
        return []
    start = int(rng.integers(0, max_start))
    Rin = R_full[start:start + T_in][:, asset_idx]
    Rout = R_full[start + T_in:start + T_in + T_out][:, asset_idx]

    common = dict(trial=trial_id, universe=universe, n_assets=n_assets, T_in=T_in)
    seeds = rng.standard_normal((M_SEEDS, n_assets))
    rows = []
    try:
        est = HierarchicalRiskParity()
        est.fit(Rin)
        w = est.weights_
        corr_in = cov_to_corr(np.asarray(est._cov_estimator.covariance_, dtype=float))
        V, D = nested_tree_factor(corr_in, est.order_, 0)  # F=0: independent reference
        Fq, Wq = _nodes(V.shape[1])

        rows.append(dict(common, iter=0, es95=es95(Rout @ w), l1_step=np.nan, note=""))
        for k in range(1, N_ITER + 1):
            theta = calibrate_factor(w, V, D, Fq, Wq, points=POINTS)
            w_new = transport_weights(theta, corr_in, seeds)
            l1_step = float(np.abs(w_new - w).sum())
            w = w_new
            rows.append(dict(common, iter=k, es95=es95(Rout @ w), l1_step=l1_step, note=""))
    except Exception as e:
        rows.append(dict(common, iter=-1, es95=None, l1_step=None,
                          note=f"{type(e).__name__}: {e}"[:120]))
    return rows


def main(n_trials: int = 500, seed: int | None = None):
    panels = {name: load_universe(name) for name in UNIVERSES}
    rng = np.random.default_rng(seed)
    write_header = not os.path.exists(OUT_CSV)
    t_start = time.time()
    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for i in range(n_trials):
            for row in run_trial(i, rng, panels):
                writer.writerow(row)
            f.flush()
            if i % 20 == 0:
                elapsed = time.time() - t_start
                print(f"trial {i}/{n_trials}  ({elapsed:.0f}s, "
                      f"{elapsed / max(i, 1):.2f}s/trial)", flush=True)
    print(f"done: {n_trials} trials in {time.time() - t_start:.0f}s -> {OUT_CSV}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    main(n)
