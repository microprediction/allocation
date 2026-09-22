"""Can a race predict what an allocator does on a sub-universe?

Take an allocator's weights on the full universe. Now restrict to a subset S
and recompute. Two ways to predict the answer without recomputing:

  rescaling    w[S] / sum(w[S]).  This is Luce's axiom, independence of
               irrelevant alternatives: removing assets leaves the odds among
               the survivors unchanged.

  the race     calibrate abilities from the full-universe weights, then run
               the race among S alone. Thurstone's model, which is not IIA:
               who wins among the survivors depends on the whole field's
               geometry, not only on their own odds.

The question is which predicts the recomputed portfolio, and whether the
answer differs by allocator. A rule that is scale-free in its own construction
should be close to IIA; one whose restriction depends on the sub-covariance
should not be.
"""
import sys
import numpy as np
import winning
from confound import hrp, min_var
from robust import random_structure, mv_lo


def factor_form(C, k=3):
    """k-factor form of a correlation, so the race is O(n k)."""
    d = np.sqrt(np.clip(np.diag(C), 1e-300, None))
    R = C / np.outer(d, d)
    ev, U = np.linalg.eigh((R + R.T) / 2)
    i = np.argsort(ev)[::-1][:k]
    V = U[:, i] * np.sqrt(np.clip(ev[i], 0.0, None))
    D = np.clip(1.0 - (V ** 2).sum(1), 1e-6, None)
    sc = np.sqrt((V ** 2).sum(1) + D)
    return V / sc[:, None], D / sc ** 2


def race_restricted(ability, idx, V=None, D=None):
    a = np.asarray(ability, float)[idx]
    if V is None:
        p = winning.race_probabilities(a)
    else:
        p = winning.race_probabilities(a, V=V[idx], D=D[idx])
    p = np.asarray(p[0] if isinstance(p, tuple) else p, float)
    return p / p.sum()


def rescaled(w, idx):
    x = np.asarray(w, float)[idx]
    return x / x.sum()


def run(n=400, m=40, T=1200, draws=40, seed=5):
    rng = np.random.default_rng(seed)
    rows = {a: {"race": [], "race+corr": [], "rescale": []}
            for a in ("HRP", "min-var", "inverse variance")}
    for _ in range(draws):
        Sig, _ = random_structure(rng, n)
        X = rng.normal(size=(T, n)) @ np.linalg.cholesky(Sig).T
        S = np.cov(X, rowvar=False)
        idx = np.sort(rng.choice(n, m, replace=False))
        Ssub = S[np.ix_(idx, idx)]
        Xsub = X[:, idx]

        full = {
            "HRP": hrp(S),
            "min-var": mv_lo(S),
            "inverse variance": (lambda v: v / v.sum())(1.0 / np.diag(S)),
        }
        truth = {
            "HRP": hrp(Ssub),
            "min-var": mv_lo(Ssub),
            "inverse variance": (lambda v: v / v.sum())(1.0 / np.diag(Ssub)),
        }
        V, D = factor_form(S, 3)
        for name, w in full.items():
            t = truth[name]
            # under independence, which is the pure choice model
            a0 = np.asarray(winning.calibrate_abilities(np.maximum(w, 1e-12)), float)
            rows[name]["race"].append(float(np.abs(race_restricted(a0, idx) - t).sum()))
            # calibrated AND restricted under the estimated correlation, which
            # is what a restriction that depends on dependence needs
            ac = np.asarray(winning.calibrate_abilities(
                np.maximum(w, 1e-12), V=V, D=D), float)
            rows[name]["race+corr"].append(
                float(np.abs(race_restricted(ac, idx, V, D) - t).sum()))
            rows[name]["rescale"].append(float(np.abs(rescaled(w, idx) - t).sum()))
    return rows


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    print(f"universe {n}, sub-universe {m}, 40 draws.")
    print("L1 error in predicting the recomputed portfolio (total weight 1.0)\n")
    rows = run(n=n, m=m)
    print(f"{'allocator':20s}{'race':>9s}{'race+corr':>12s}{'rescaling':>12s}"
          f"{'best':>14s}")
    for name, d in rows.items():
        r = np.asarray(d["race"]); c = np.asarray(d["race+corr"]); q = np.asarray(d["rescale"])
        meds = {"race": np.median(r), "race+corr": np.median(c), "rescaling": np.median(q)}
        best = min(meds, key=meds.get)
        print(f"{name:20s}{meds['race']:9.4f}{meds['race+corr']:12.4f}"
              f"{meds['rescaling']:12.4f}{best:>14s}")
