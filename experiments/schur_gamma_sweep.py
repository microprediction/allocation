"""Follow-up to schur_thurstone_repair.py: does the right amount of repair
(max_factors) shrink as Schur's own gamma rises?

The first result showed something worth checking rather than assuming: for
HRP (gamma=0) the gauss-repair improved monotonically out to F=5, but for
Schur(gamma=0.5) it PEAKED at F=2 and got worse beyond that -- i.e. too much
repair started double-counting cross-block covariance Schur's own gamma
already used. If that's real, the empirically-best F should fall as gamma
rises (gamma=1 needing ~F=0, since an exact min-variance Schur has no
missing second-moment covariance left to restore -- the point raised when
this was first proposed).

This sweeps gamma in {0, 0.25, 0.5, 0.75, 1.0} x a small F grid, gauss-repair
only (no tail race -- isolates the correlation-completion question), and
reports the best F per gamma, averaged over the same style of bowling
markets as the main experiment.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from bowling_sim import generate
from schur_thurstone_repair import _nodes, es95, nested_tree_factor

from allocation import SchurComplementary
from allocation._thurstone.covariance import cov_to_corr
from allocation._thurstone.factor import calibrate_factor
from allocation._thurstone.transport import transport_weights

GAMMA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
FACTOR_GRID = [0, 1, 2, 3, 5]
M_SEEDS = 1 << 14


def run_one(seed, n=20, T=1000, k=3, sd=6.0, rng_seed=123):
    R, *_ = generate(n=n, T=T, seed=seed, k=k, sd=sd)
    Tin = T // 2
    Rin, Rout = R[:Tin], R[Tin:]
    rng = np.random.default_rng(rng_seed)
    seeds = rng.standard_normal((M_SEEDS, n))

    out = {}
    for gamma in GAMMA_GRID:
        est = SchurComplementary(gamma=gamma, keep_monotonic=False)
        est.fit(Rin)
        w_base = est.weights_
        order = est.order_
        corr_in = cov_to_corr(np.asarray(est._cov_estimator.covariance_, dtype=float))
        base_es = es95(Rout @ w_base)
        row = {}
        for mf in FACTOR_GRID:
            V, D = nested_tree_factor(corr_in, order, mf)
            F, W = _nodes(V.shape[1])
            theta = calibrate_factor(w_base, V, D, F, W, points=251)
            w_g = transport_weights(theta, corr_in, seeds)
            row[mf] = es95(Rout @ w_g)
        out[gamma] = (base_es, row)
    return out


def main():
    configs = [(s, sd) for s in range(4) for sd in (4.0, 6.0, 8.0, 10.0)]
    agg = {g: {"base": []} for g in GAMMA_GRID}
    for g in GAMMA_GRID:
        for mf in FACTOR_GRID:
            agg[g][mf] = []

    for seed, sd in configs:
        result = run_one(seed, sd=sd)
        for g, (base_es, row) in result.items():
            agg[g]["base"].append(base_es)
            for mf, v in row.items():
                agg[g][mf].append(v)

    print(f"markets: {len(configs)}\n")
    print(f"{'gamma':>7}{'base':>10}" + "".join(f"{'F='+str(mf):>9}" for mf in FACTOR_GRID)
          + f"{'best F':>9}{'best ES95':>12}")
    best_f_by_gamma = {}
    for g in GAMMA_GRID:
        base = float(np.mean(agg[g]["base"]))
        vals = {mf: float(np.mean(agg[g][mf])) for mf in FACTOR_GRID}
        best_mf = max(FACTOR_GRID, key=lambda mf: vals[mf])
        best_f_by_gamma[g] = best_mf
        print(f"{g:>7.2f}{base:>10.4f}" + "".join(f"{vals[mf]:>9.4f}" for mf in FACTOR_GRID)
              + f"{best_mf:>9}{vals[best_mf]:>12.4f}")

    print("\nbest F by gamma:", best_f_by_gamma)
    fs = [best_f_by_gamma[g] for g in GAMMA_GRID]
    monotone = all(fs[i] >= fs[i + 1] for i in range(len(fs) - 1))
    print(f"non-increasing in gamma: {monotone}")


if __name__ == "__main__":
    main()
