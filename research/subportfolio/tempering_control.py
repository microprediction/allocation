"""Does the race know anything, or is it only de-concentrating?

The index study found equal weight tying the race and read that as the race
adding nothing. That reading is confounded. Restricting a cap-weighted parent
leaves a portfolio far more concentrated than the sub-universe's own optimum:
15 effective names against the oracle's 73, on 200 assets. When the parent is
over-confident in that way, ANY move toward equal weight is paid, and equal
weight moves furthest. It collects the reward without knowing anything.

So equal weight is the wrong control. The right one holds concentration fixed
and changes only which names carry the weight. Tempering does that:

    w(beta) proportional to parent^beta,   beta in [0, 1]

beta = 1 is proportional restriction, beta = 0 is equal weight, and beta is
solved by bisection so the tempered portfolio has the SAME effective number of
names as the race. Any variance difference that remains is about name
selection, because the concentration is identical by construction.

Three baselines at the race's own concentration:

    tempered     parent^beta, so it keeps the parent's ORDER
    shuffled     the same weight vector on a random permutation of the names,
                 which keeps the concentration and destroys the information
    oracle       the long-only optimum, for the room available

If the race beats tempered, it is choosing names better than the parent's own
ordering does. If it only beats shuffled, it has the parent's ordering and
nothing more. If it beats neither, de-concentration was the whole story.
"""
import sys
import time

import numpy as np

from markets import IndexMarket
from rules import long_only_min_var, factor_correlation, proportional, race


def eff(w):
    return 1.0 / float(np.sum(np.asarray(w, float) ** 2))


def temper(w, target_eff, lo=0.0, hi=1.0, iters=60):
    """parent^beta renormalised, with beta solved so eff() hits the target."""
    w = np.asarray(w, float)
    lw = np.log(np.maximum(w, 1e-300))

    def at(b):
        x = np.exp(b * (lw - lw.max()))
        return x / x.sum()

    if eff(at(hi)) >= target_eff:          # even beta=1 is diffuse enough
        return at(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if eff(at(mid)) > target_eff:
            lo = mid
        else:
            hi = mid
    return at(0.5 * (lo + hi))


def run(n=2000, m=200, draws=10, seed=31, T=104, k=2):
    rng = np.random.default_rng(seed)
    keys = ("proportional", "race", "race+factor", "tempered", "tempered+factor",
            "shuffled", "equal weight", "oracle")
    out = {key: [] for key in keys}
    effs = {key: [] for key in keys}
    for d in range(draws):
        mk = IndexMarket(rng, n, rank=5, target_corr=0.27)
        idx = np.sort(rng.choice(n, m, replace=False))
        Sub = mk.block(idx)
        v = lambda w: float(np.asarray(w, float) @ Sub @ np.asarray(w, float))
        X = mk.panel(rng, T)
        _, V, D = factor_correlation(X, k)

        pw = proportional(mk.parent, idx)
        rw = race(mk.parent, idx)[0]
        rf = race(mk.parent, idx, V=V, D=D)[0]
        tw = temper(pw, eff(rw))                 # matched to the plain race
        tf = temper(pw, eff(rf))                 # matched to race+factor
        sh = tw[rng.permutation(m)]
        cand = {"proportional": pw, "race": rw, "race+factor": rf,
                "tempered": tw, "tempered+factor": tf, "shuffled": sh,
                "equal weight": np.full(m, 1.0 / m),
                "oracle": long_only_min_var(Sub)}
        for key, w in cand.items():
            out[key].append(v(w))
            effs[key].append(eff(w))
        print(f"  ...draw {d + 1}/{draws}", flush=True)
    return out, effs, keys


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    t0 = time.time()
    out, effs, keys = run(n=n, m=m)
    base = np.array(out["proportional"])
    nd = len(base)
    print(f"\nparent {n}, sub-universe {m}, {nd} draws. Ratio to proportional, "
          f"with the effective\nnumber of names each rule holds. Tempered rows "
          f"are matched to the race's concentration.\n")
    print(f"{'rule':20s}{'eff names':>11}{'vs proportional':>17}")
    for key in keys:
        print(f"{key:20s}{np.median(effs[key]):11.0f}"
              f"{np.median(np.array(out[key]) / base):17.3f}")

    print(f"\nthe comparisons that control for concentration\n")
    for a, b in (("race", "tempered"), ("race+factor", "tempered+factor"),
                 ("race", "shuffled"), ("tempered", "shuffled"),
                 ("race+factor", "equal weight")):
        x, y = np.array(out[a]), np.array(out[b])
        print(f"  {a:16s} < {b:18s}{np.mean(x < y):5.0%} of draws   "
              f"median ratio {np.median(x / y):.4f}")
    print(f"\n{time.time() - t0:.0f}s")
