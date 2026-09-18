"""Does the repair's benefit depend on THIS package's Fiedler seriation, or
does it generalize to classical (Lopez de Prado 2016) single-linkage HRP?

Every other experiment in this line of work reads `order_` straight off the
fitted `HierarchicalRiskParity`/`SchurComplementary` estimator, which orders
assets by the Fiedler vector of the correlation-affinity graph -- this
package's own smooth-turnover upgrade, not the original dendrogram order. This
checks whether that choice is load-bearing: for the SAME in-sample
covariance, build both a Fiedler order and a classical single-linkage
quasi-diagonal order (the standard HRP construction, implemented fresh here
since this package only ships the Fiedler version), compute HRP weights and
the repair under EACH, and compare out-of-sample ES95 gains.

Real S&P 500 / REIT sub-portfolios, gamma=0 (plain HRP) only -- the classical
method's own name -- since the question is about seriation, not the Schur
coupling. Results appended to data/seriation_check_results.csv.
"""
from __future__ import annotations

import csv
import os
import time
import warnings

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

warnings.filterwarnings("ignore")

from schur_thurstone_repair import _nodes, es95, nested_tree_factor

from allocation._schur.coupling import compute_weights
from allocation._schur.seriation import seriate
from allocation._thurstone.covariance import cov_to_corr
from allocation._thurstone.factor import calibrate_factor
from allocation._thurstone.transport import transport_weights
from overnight_repair_validation import UNIVERSES, load_universe

F_GRID = [0, 1, 2]
M_SEEDS = 1 << 14
POINTS = 251
OUT_CSV = "data/seriation_check_results.csv"
FIELDS = ["trial", "universe", "n_assets", "T_in", "T_out", "start_idx",
          "seriation", "F", "base_es95", "repair_es95", "gain", "note"]


def classical_hrp_order(corr: np.ndarray) -> np.ndarray:
    """Lopez de Prado (2016) quasi-diagonal order: single-linkage clustering
    of the correlation distance d_ij = sqrt(0.5*(1 - corr_ij)), then the
    standard recursive dendrogram-to-leaf-order expansion."""
    n = corr.shape[0]
    d = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(d, 0.0)
    link = sch.linkage(ssd.squareform(d, checks=False), method="single")
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i, j = df0.index, df0.values - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    assert sort_ix.max() == num_items - 1 and len(sort_ix) == n
    return sort_ix.to_numpy()


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

    cov_in = np.cov(Rin, rowvar=False)
    corr_in = cov_to_corr(cov_in)
    seeds = rng.standard_normal((M_SEEDS, n_assets))
    common = dict(trial=trial_id, universe=universe, n_assets=n_assets, T_in=T_in,
                  T_out=T_out, start_idx=start)

    order_fiedler, _ = seriate(cov_in)
    order_classical = classical_hrp_order(corr_in)

    rows = []
    for name, order in (("fiedler", order_fiedler), ("classical", order_classical)):
        t0 = time.time()
        try:
            w_base = compute_weights(order, cov_in, gamma=0.0)
            base_es = es95(Rout @ w_base)
            for F in F_GRID:
                V, D = nested_tree_factor(corr_in, order, F)
                Fq, Wq = _nodes(V.shape[1])
                theta = calibrate_factor(w_base, V, D, Fq, Wq, points=POINTS)
                w_rep = transport_weights(theta, corr_in, seeds)
                rep_es = es95(Rout @ w_rep)
                rows.append(dict(common, seriation=name, F=F, base_es95=base_es,
                                  repair_es95=rep_es, gain=rep_es - base_es, note=""))
        except Exception as e:
            rows.append(dict(common, seriation=name, F=-1, base_es95=None,
                              repair_es95=None, gain=None,
                              note=f"{type(e).__name__}: {e}"[:120]))
    return rows


def main(n_trials: int = 800, seed: int | None = None):
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
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    main(n)
