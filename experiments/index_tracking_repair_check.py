"""Does the repair help a REAL cap-weighted index, not just an allocator's
output?

Every other check in this line of work repairs the weights some ALLOCATOR
produced (HRP, inverse-variance, minimum-variance...). This is the
motivating case from the companion Thurstone-portfolios paper itself: an
actual historical S&P 500 constituent weight snapshot, restricted to a
random sub-universe (so it is renormalized, not the true index weight --
exactly the "restricted-universe cap weighting is not the efficient
restricted portfolio" critique that paper opens with), calibrated under
independence (F=0), and re-raced under the trailing correlation. Out-of-
sample is the ~60 trading days AFTER the snapshot date (roughly the next
monthly rebalance).

Cap-weight source: S&P 500 constituent weights from winningportstudy
(data/s500_wt.csv, ~monthly snapshots, 2014-2024), matched against the same
return panel used everywhere else in this paper
(data/sp500_returns_2014_2024.parquet). Build the weight snapshot table
first if it doesn't exist yet: see the inline block at the bottom of this
file, or read it off the parquet cache directly.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from schur_thurstone_repair import _nodes, es95

from allocation._thurstone.factor import calibrate_factor
from allocation._thurstone.transport import transport_weights

M_SEEDS = 1 << 14
POINTS = 251
T_OUT = 60


def independent_repair(w_base, corr_full, seeds):
    n = len(w_base)
    V = np.zeros((n, 1))
    D = np.ones(n)
    Fq, Wq = _nodes(1)
    theta = calibrate_factor(w_base, V, D, Fq, Wq, points=POINTS)
    return transport_weights(theta, corr_full, seeds)


def load_data():
    ret = pd.read_parquet("data/sp500_returns_2014_2024.parquet")
    ret.columns = ret.columns.astype(str)
    wt = pd.read_parquet("data/sp500_capweights_2014_2024.parquet")
    wt.columns = wt.columns.astype(str)
    return ret, wt


def run_trial(rng, ret, wt, min_gap=260, min_lead=250):
    dates = ret.index.to_numpy()
    n_dates = len(dates)
    # a snapshot date needs min_lead trading days before it (correlation
    # window) and T_OUT after it (out-of-sample), and must itself be a
    # cap-weight snapshot date
    valid_snap_dates = [d for d in wt.index if d in ret.index]
    lo = dates[min_lead]
    hi = dates[n_dates - T_OUT - 1]
    valid_snap_dates = [d for d in valid_snap_dates if lo <= d <= hi]
    if not valid_snap_dates:
        return None
    snap = valid_snap_dates[rng.integers(0, len(valid_snap_dates))]
    pos = int(np.searchsorted(dates, snap))

    row = wt.loc[snap].dropna()
    row = row[row > 0]
    if len(row) < 12:
        return None
    n_assets = int(rng.integers(12, min(30, len(row)) + 1))
    ids = rng.choice(row.index.to_numpy(), size=n_assets, replace=False)

    w_cap = row[ids].to_numpy(dtype=float)
    w_cap = w_cap / w_cap.sum()

    Rin = ret[ids].to_numpy()[max(0, pos - min_lead):pos]
    Rout = ret[ids].to_numpy()[pos:pos + T_OUT]
    if len(Rin) < 100 or len(Rout) < T_OUT:
        return None

    corr_in = np.corrcoef(Rin, rowvar=False)
    seeds = rng.standard_normal((M_SEEDS, n_assets))
    try:
        w_rep = independent_repair(w_cap, corr_in, seeds)
    except Exception:
        return None

    r_cap = Rout @ w_cap
    r_rep = Rout @ w_rep
    return (es95(r_rep) - es95(r_cap), float(r_rep.mean() - r_cap.mean()),
            float(np.abs(w_rep - w_cap).sum()))


def main(n_trials=300, seed=0):
    ret, wt = load_data()
    print(f"returns: {ret.shape}, cap weights: {wt.shape}", flush=True)
    rng_master = np.random.default_rng(seed)
    es_gains, mean_gains, turnovers = [], [], []
    t0 = time.time()
    for i in range(n_trials):
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        out = run_trial(rng, ret, wt)
        if out is not None:
            es_g, mean_g, l1 = out
            es_gains.append(es_g)
            mean_gains.append(mean_g)
            turnovers.append(l1)
        if i % 25 == 0:
            print(f"{i}/{n_trials} ({time.time()-t0:.0f}s)", flush=True)
    es_gains = np.array(es_gains)
    mean_gains = np.array(mean_gains)
    turnovers = np.array(turnovers)
    print(f"\ndone in {time.time()-t0:.0f}s, n={len(es_gains)}\n")
    print(f"ES95 mean gain:  {es_gains.mean():.5f}   win rate: {100*(es_gains>0).mean():.1f}%")
    print(f"return win rate: {100*(mean_gains>0).mean():.1f}%")
    print(f"mean |w_repair - w_cap|_1 (turnover from the raw index weight): {turnovers.mean():.4f}")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
