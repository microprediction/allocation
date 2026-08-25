"""Does the repair survive an ESTIMATED (not known) covariance?

Every earlier run built C_used and the repair target from `cov_estimator_.
covariance_` fit on the in-sample window -- always an estimate, never an
oracle -- but always with a fairly generous window (T_in=500 for n=20, a
25:1 ratio). HRP's own motivating claim is specifically robustness to
estimation error, so the repair needs to be checked where that error is
large, not just where the in-sample covariance is already good.

Fixes the OUT-OF-SAMPLE window (last 500 rows, identical across runs) and
shrinks T_in from a generous 30x the asset count down to a bare 2x, at F=1
and F=2 (the levels that were doing most of the work in the main sweep),
for both HRP and Schur(gamma=0.5).
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from bowling_sim import generate
from schur_thurstone_repair import _nodes, es95, nested_tree_factor

from allocation import HierarchicalRiskParity, SchurComplementary
from allocation._thurstone.covariance import cov_to_corr
from allocation._thurstone.factor import calibrate_factor
from allocation._thurstone.transport import transport_weights

TIN_GRID = [40, 80, 150, 300, 600]
FACTOR_LEVELS = [1, 2]
T_OUT = 500
M_SEEDS = 1 << 14


def run_one(seed, n=20, k=3, sd=6.0, rng_seed=123):
    T_total = max(TIN_GRID) + T_OUT
    R, *_ = generate(n=n, T=T_total, seed=seed, k=k, sd=sd)
    Rout = R[-T_OUT:]                                    # FIXED across Tin
    rng = np.random.default_rng(rng_seed)
    seeds = rng.standard_normal((M_SEEDS, n))

    out = {}
    for Tin in TIN_GRID:
        Rin = R[max(TIN_GRID) - Tin: max(TIN_GRID)]       # most recent Tin rows before Rout
        for name, est in (
            ("hrp", HierarchicalRiskParity()),
            ("schur(g=0.5)", SchurComplementary(gamma=0.5, keep_monotonic=False)),
        ):
            est.fit(Rin)
            w_base = est.weights_
            order = est.order_
            corr_in = cov_to_corr(np.asarray(est._cov_estimator.covariance_, dtype=float))
            base_es = es95(Rout @ w_base)
            row = {"base": base_es}
            for mf in FACTOR_LEVELS:
                V, D = nested_tree_factor(corr_in, order, mf)
                F, W = _nodes(V.shape[1])
                theta = calibrate_factor(w_base, V, D, F, W, points=251)
                w_g = transport_weights(theta, corr_in, seeds)
                row[mf] = es95(Rout @ w_g)
            out[(name, Tin)] = row
    return out


def main():
    configs = [(s, sd) for s in range(4) for sd in (5.0, 7.0, 9.0)]
    agg = {}
    for seed, sd in configs:
        result = run_one(seed, sd=sd)
        for key, row in result.items():
            slot = agg.setdefault(key, {})
            for k2, v in row.items():
                slot.setdefault(k2, []).append(v)

    print(f"markets: {len(configs)}  (T_out fixed at {T_OUT}, T_in varies)\n")
    for name in ("hrp", "schur(g=0.5)"):
        print(f"--- {name} ---")
        print(f"{'T_in':>6}{'n/T_in':>8}{'base':>10}"
              + "".join(f"{'F='+str(mf):>10}" for mf in FACTOR_LEVELS)
              + f"{'best gain':>12}")
        for Tin in TIN_GRID:
            row = agg[(name, Tin)]
            base = float(np.mean(row["base"]))
            vals = {mf: float(np.mean(row[mf])) for mf in FACTOR_LEVELS}
            best_gain = max(vals.values()) - base
            print(f"{Tin:>6}{20 / Tin:>8.2f}{base:>10.4f}"
                  + "".join(f"{vals[mf]:>10.4f}" for mf in FACTOR_LEVELS)
                  + f"{best_gain:>12.4f}")
        print()


if __name__ == "__main__":
    main()
