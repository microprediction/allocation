"""Does tail-dependence repair help during an ACTUAL historical crash, not
just averaged over mostly-quiet random windows?

Every real-data check so far drew a RANDOM 1-year out-of-sample window from
2014-2024. Most such windows are quiet; a rare tail-dependence effect that
only shows up during genuine systemic stress gets diluted to nothing when
averaged against years of normal markets, and ES99 on a random year still
only samples ~2-3 "bad" days that need not be a real crash at all. This
tests directly against the one unambiguous, real crash in the data range:
COVID, Feb 19 - Mar 23 2020 (23 trading days; worst equal-weighted market
day in that window is -14.2%, confirmed extreme).

In-sample window ends the day before the crash starts (no lookahead);
out-of-sample IS the crash window itself, fixed across trials (only the
random sub-universe and in-sample length vary). With only 23 OOS days,
per-trial quantile statistics (ES95/ES99) are too noisy to be meaningful;
mean return over the crash window is the metric, pooled across many
trials/sub-universes for power.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from schur_thurstone_repair import _nodes, nested_tree_factor

from allocation import HierarchicalRiskParity
from allocation._thurstone.covariance import cov_to_corr
from allocation._thurstone.factor import calibrate_factor
from allocation._thurstone.transport import race_weights, transport_weights

COVID_START = 20200219
COVID_END = 20200323


def load():
    ret = pd.read_parquet("data/sp500_returns_2014_2024.parquet")
    dates = ret.index.to_numpy()
    pos_start = int(np.searchsorted(dates, COVID_START))
    pos_end = int(np.searchsorted(dates, COVID_END))
    return ret, pos_start, pos_end


def run_trial(rng, ret_arr, pos_start, pos_end, N_total, F):
    n_assets = int(rng.integers(12, 30))
    asset_idx = rng.choice(N_total, size=n_assets, replace=False)
    T_in = int(rng.integers(250, 750))
    if pos_start - T_in < 0:
        return None
    Rin = ret_arr[pos_start - T_in:pos_start][:, asset_idx]
    Rout = ret_arr[pos_start:pos_end][:, asset_idx]  # the crash itself, fixed

    est = HierarchicalRiskParity(); est.fit(Rin)
    w_base = est.weights_
    corr_in = cov_to_corr(np.asarray(est._cov_estimator.covariance_, dtype=float))
    V, D = nested_tree_factor(corr_in, est.order_, F)
    Fq, Wq = _nodes(V.shape[1])
    theta = calibrate_factor(w_base, V, D, Fq, Wq, points=251)

    med = np.median(Rin, axis=0, keepdims=True)
    mad = np.median(np.abs(Rin - med), axis=0) * 1.4826
    mad = np.maximum(mad, 1e-8)
    Lstd_in = -(Rin - med) / mad
    boot_idx = rng.integers(0, T_in, 1 << 14)
    seeds = rng.standard_normal((1 << 14, n_assets))

    w_gauss = transport_weights(theta, corr_in, seeds)
    w_hist = race_weights(theta[None, :] + Lstd_in[boot_idx])

    r_base = (Rout @ w_base).mean()
    r_gauss = (Rout @ w_gauss).mean()
    r_hist = (Rout @ w_hist).mean()
    worst_day_base = (Rout @ w_base).min()
    worst_day_gauss = (Rout @ w_gauss).min()
    worst_day_hist = (Rout @ w_hist).min()
    return (r_gauss - r_base, r_hist - r_base, r_hist - r_gauss,
            worst_day_gauss - worst_day_base, worst_day_hist - worst_day_base)


def main(n_trials=500, F=0, seed=0):
    ret, pos_start, pos_end = load()
    ret_arr = ret.to_numpy()
    N_total = ret_arr.shape[1]
    print(f"crash window: rows {pos_start}:{pos_end} ({pos_end-pos_start} days)", flush=True)
    rows = []
    rng_master = np.random.default_rng(seed)
    t0 = time.time()
    for i in range(n_trials):
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        out = run_trial(rng, ret_arr, pos_start, pos_end, N_total, F)
        if out is not None:
            rows.append(out)
        if i % 100 == 0:
            print(f"{i}/{n_trials} ({time.time()-t0:.0f}s)", flush=True)
    rows = np.array(rows)
    gauss_g, hist_g, hist_vs_gauss, wd_gauss, wd_hist = rows.T
    print(f"\ndone in {time.time()-t0:.0f}s, n={len(rows)}\n")
    print(f"crash-window MEAN return, gauss repair vs base:  mean={gauss_g.mean():.5f}  win%={100*(gauss_g>0).mean():.1f}")
    print(f"crash-window MEAN return, hist  repair vs base:  mean={hist_g.mean():.5f}  win%={100*(hist_g>0).mean():.1f}")
    print(f"crash-window MEAN return, hist vs GAUSS (incremental tail effect):")
    print(f"                                                  mean={hist_vs_gauss.mean():.5f}  win%={100*(hist_vs_gauss>0).mean():.1f}")
    print(f"\nWORST SINGLE DAY, gauss repair vs base:          mean={wd_gauss.mean():.5f}  win%={100*(wd_gauss>0).mean():.1f}")
    print(f"WORST SINGLE DAY, hist  repair vs base:          mean={wd_hist.mean():.5f}  win%={100*(wd_hist>0).mean():.1f}")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
