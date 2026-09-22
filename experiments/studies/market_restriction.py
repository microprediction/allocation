"""Leveraging a market-found optimum onto sub-universes.

Assume with CAPM that the cap-weighted market portfolio is optimal. Then the
market weights encode an optimum nobody had to estimate: the collective
result, not a sample covariance. The question is whether that optimality can
be transported to a sub-universe.

The predictor gets the market weights and NOTHING else. No covariance, no
returns. That is the situation of someone holding an index and wanting the
right portfolio on a sector, a screen or an exclusion list.

  rescaling   renormalise the market weights on the survivors. This is what a
              cap-weighted sector fund does, and it is Luce's axiom.
  the race    calibrate abilities that reproduce the market weights, then race
              among the survivors alone. Thurstone, which is not IIA.

Truth is the optimal portfolio on the sub-universe under the TRUE covariance,
so there is no estimation anywhere and the comparison is purely about the
restriction map.
"""
import sys
import numpy as np
import cvxpy as cp
import winning
from robust import random_structure


def optimum(Sig, idx=None):
    """Long-only minimum variance on the true covariance, solved."""
    C = Sig if idx is None else Sig[np.ix_(idx, idx)]
    m = C.shape[0]
    x = cp.Variable(m)
    cp.Problem(cp.Minimize(cp.quad_form(x, cp.psd_wrap(C))),
               [cp.sum(x) == 1, x >= 0]).solve(solver=cp.CLARABEL)
    w = np.maximum(x.value, 0.0)
    return w / w.sum()


def run(n=150, keep_frac=0.4, draws=40, seed=3):
    rng = np.random.default_rng(seed)
    out = {"rescale": [], "race": [], "equal": []}
    for _ in range(draws):
        Sig, _ = random_structure(rng, n)
        market = optimum(Sig)                       # the market found this
        m = int(round(keep_frac * n))
        idx = np.sort(rng.choice(n, m, replace=False))
        truth = optimum(Sig, idx)                   # what we want, never seen

        resc = market[idx] / market[idx].sum()
        a = np.asarray(winning.calibrate_abilities(np.maximum(market, 1e-12)), float)
        p = winning.race_probabilities(a[idx])
        p = np.asarray(p[0] if isinstance(p, tuple) else p, float)
        race = p / p.sum()

        out["rescale"].append(float(np.abs(resc - truth).sum()))
        out["race"].append(float(np.abs(race - truth).sum()))
        out["equal"].append(float(np.abs(np.full(m, 1 / m) - truth).sum()))
    return {k: np.asarray(v) for k, v in out.items()}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    print(f"universe {n}, true covariance, no estimation anywhere.")
    print("Predicting the sub-universe optimum from the market weights alone.\n")
    print(f"{'keep':>7s}{'equal wt':>11s}{'rescaling':>12s}{'race':>10s}"
          f"{'race beats rescaling':>24s}")
    for frac in (0.7, 0.5, 0.3, 0.15):
        r = run(n=n, keep_frac=frac)
        w = float(np.mean(r["race"] < r["rescale"]))
        z = 1.96; nn = len(r["race"]); dd = 1+z*z/nn
        c = (w+z*z/(2*nn))/dd; h = z*((w*(1-w)/nn+z*z/(4*nn*nn))**0.5)/dd
        print(f"{frac:7.0%}{np.median(r['equal']):11.4f}{np.median(r['rescale']):12.4f}"
              f"{np.median(r['race']):10.4f}"
              f"{f'{w:.0%} [{max(c-h,0):.0%},{min(c+h,1):.0%}]':>24s}")
