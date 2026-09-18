"""Overnight robustness validation: real equity/REIT sub-portfolios, varied
estimator and configuration choices.

Each trial draws, independently at random: a universe (S&P 500 constituents
2014-2024, or the REIT sector 2000-2024 -- both real return panels, sourced
from https://github.com/microprediction/winningportstudy), a sub-portfolio
size and random asset subset from that universe, a random in-sample/
out-of-sample window, a covariance estimator (the package's own EwmaCovariance
at two halflives, or precise's LedoitWolfCovariance/OASCovariance/
EmpiricalCovariance), and Schur's coupling gamma. Within that one
configuration, F is SWEPT over a small grid (not drawn once), so each trial
directly reproduces the depth-dependence structure of the synthetic-market
study on real data. For every F: fit HRP and Schur-complementary on the
in-sample covariance, build the nested-tree C_used at that budget
(schur_thurstone_repair.nested_tree_factor), and repair toward two targets:
  "gauss" -- the full in-sample correlation (correlation completion alone,
             the result validated on the synthetic bowling market);
  "hist"  -- the true in-sample returns, bootstrap-resampled (genuine
             empirical dependence, including whatever real tail co-movement
             is actually there -- the open question the bowling-market study
             could not confirm; real equities are far less pathologically
             leptokurtic than that synthetic generator, but the race is
             still run on losses with a robust median/MAD standardization,
             per the pitfall recorded in that study).
Out-of-sample ES95 is compared for base vs. each repair.

Results are appended to data/overnight_results.csv as each trial completes
(crash-safe, and inspectable mid-run):
    python3 -c "import pandas as pd; df = pd.read_csv('data/overnight_results.csv'); \
    print(df.groupby(['universe','method','target','F'])['gain'] \
    .agg(['mean','count', lambda s: (s>0).mean()]))"

Run from experiments/: `python3 overnight_repair_validation.py [n_trials]`.
Safe to stop and resume: re-running just appends more trials to the same CSV
(trial ids restart at 0 each run, so tag by mtime/run if you need to
distinguish runs later).
"""
from __future__ import annotations

import csv
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from schur_thurstone_repair import _nodes, es95, nested_tree_factor

from allocation import HierarchicalRiskParity, SchurComplementary
from allocation._thurstone.covariance import cov_to_corr
from allocation._thurstone.factor import calibrate_factor
from allocation._thurstone.transport import race_weights, transport_weights
from allocation.moments import EwmaCovariance
from precise import EmpiricalCovariance, LedoitWolfCovariance, OASCovariance

UNIVERSES = {
    "sp500": "data/sp500_returns_2014_2024.parquet",
    "reit": "data/reit_returns.parquet",
}
EST_CHOICES = ["ewma_short", "ewma_long", "ledoitwolf", "oas", "empirical"]
GAMMA_CHOICES = [0.0, 0.25, 0.5, 0.75, 1.0]
F_GRID = [0, 1, 2, 3]
M_SEEDS = 1 << 15
POINTS = 401

OUT_CSV = "data/overnight_results.csv"
FIELDS = [
    "trial", "universe", "n_assets", "T_in", "T_out", "start_idx", "est", "gamma", "F",
    "method", "target", "base_es95", "repair_es95", "gain", "elapsed_s", "note",
]


def make_estimator(name: str):
    if name == "ewma_short":
        return EwmaCovariance(halflife=20.0)
    if name == "ewma_long":
        return EwmaCovariance(halflife=120.0)
    if name == "ledoitwolf":
        return LedoitWolfCovariance()
    if name == "oas":
        return OASCovariance()
    if name == "empirical":
        return EmpiricalCovariance()
    raise ValueError(name)


def load_universe(name: str):
    df = pd.read_parquet(UNIVERSES[name])
    return df.values.astype(float), list(df.columns)


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

    est_name = str(rng.choice(EST_CHOICES))
    gamma = float(rng.choice(GAMMA_CHOICES))

    # loss-side, robust (median/MAD) standardization of the in-sample returns
    # -- see the "centering pitfall" remark in the companion paper; raw std
    # is not safe even for real (far less pathological) equity returns.
    med = np.median(Rin, axis=0, keepdims=True)
    mad = np.median(np.abs(Rin - med), axis=0) * 1.4826
    mad = np.maximum(mad, 1e-8)
    Lstd_in = -(Rin - med) / mad
    boot_idx = rng.integers(0, T_in, M_SEEDS)
    seeds = rng.standard_normal((M_SEEDS, n_assets))

    common = dict(trial=trial_id, universe=universe, n_assets=n_assets, T_in=T_in,
                   T_out=T_out, start_idx=start, est=est_name, gamma=gamma)
    rows = []
    for method_name, ctor in (
        ("hrp", lambda: HierarchicalRiskParity(covariance_estimator=make_estimator(est_name))),
        ("schur", lambda: SchurComplementary(
            gamma=gamma, keep_monotonic=False, covariance_estimator=make_estimator(est_name))),
    ):
        t0 = time.time()
        try:
            est = ctor()
            est.fit(Rin)
            w_base = est.weights_
            order = est.order_
            corr_in = cov_to_corr(np.asarray(est._cov_estimator.covariance_, dtype=float))
            base_es = es95(Rout @ w_base)
            for F in F_GRID:
                V, D = nested_tree_factor(corr_in, order, F)
                Fq, Wq = _nodes(V.shape[1])
                theta = calibrate_factor(w_base, V, D, Fq, Wq, points=POINTS)
                w_gauss = transport_weights(theta, corr_in, seeds)
                w_hist = race_weights(theta[None, :] + Lstd_in[boot_idx])
                for target, w_rep in (("gauss", w_gauss), ("hist", w_hist)):
                    rep_es = es95(Rout @ w_rep)
                    rows.append(dict(common, F=F, method=method_name, target=target,
                                      base_es95=base_es, repair_es95=rep_es,
                                      gain=rep_es - base_es,
                                      elapsed_s=time.time() - t0, note=""))
        except Exception as e:  # keep the overnight run alive across bad draws
            rows.append(dict(common, F=-1, method=method_name, target="ERROR",
                              base_es95=None, repair_es95=None, gain=None,
                              elapsed_s=time.time() - t0,
                              note=f"{type(e).__name__}: {e}"[:120]))
    return rows


def main(n_trials: int = 1000, seed: int | None = None):
    panels = {name: load_universe(name) for name in UNIVERSES}
    for name, (R, cols) in panels.items():
        print(f"{name}: {R.shape[0]} days x {R.shape[1]} assets", flush=True)
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
            if i % 10 == 0:
                elapsed = time.time() - t_start
                print(f"trial {i}/{n_trials}  ({elapsed:.0f}s elapsed, "
                      f"{elapsed / max(i, 1):.1f}s/trial)", flush=True)
    print(f"done: {n_trials} trials in {time.time() - t_start:.0f}s -> {OUT_CSV}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    main(n)
