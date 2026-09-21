"""HRP's actual home: fewer observations than assets, sample covariance singular.

The claim for HRP is that it needs no inverse. So the fair question is not
whether an optimizer on the RAW sample beats it, since that object does not
exist here, but whether an optimizer on an estimate that IS invertible beats
it. Two such estimates need no tuning and no oracle:

  Ledoit-Wolf   always positive definite by construction
  block taper   blocks of size n/K are invertible whenever n/K < T, which is
                the mechanism hierarchical methods actually exploit

Inverse variance is included because it also needs no inverse, and if HRP
cannot beat it then the hierarchy is earning nothing in its own regime.
"""
import numpy as np
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import fcluster
from confound import hrp, min_var, tree
from robust import random_structure, mv_lo, clusters, nco, taper

RATIOS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def run(draws=260, seed=23):
    rng = np.random.default_rng(seed)
    out = {r: {k: [] for k in ("hrp", "nco", "ivp", "lw", "taper", "blockdiag")}
           for r in RATIOS}
    for _ in range(draws):
        n = int(rng.choice([40, 100]))
        Sig, fam = random_structure(rng, n)
        L = np.linalg.cholesky(Sig)
        for r in RATIOS:
            T = max(int(round(r * n)), 3)
            Z = rng.normal(size=(T, n))
            X = Z @ L.T
            S = np.cov(X, rowvar=False)
            # K chosen so that clusters are smaller than the sample: the only
            # data-driven choice here, and it is available to any optimizer.
            K = max(2, min(n // 2, int(np.ceil(n / max(T - 1, 1)))))
            cl = clusters(S, min(K, n))
            LW = LedoitWolf(assume_centered=False).fit(X).covariance_
            iv = 1 / np.diag(S); iv /= iv.sum()
            v = lambda w: float(w @ Sig @ w)
            out[r]["hrp"].append(v(hrp(S)))
            out[r]["nco"].append(v(nco(S, cl, n)))
            out[r]["ivp"].append(v(iv))
            out[r]["lw"].append(v(mv_lo(LW)))
            out[r]["taper"].append(v(mv_lo(taper(S, cl, 0.5, n))))
            out[r]["blockdiag"].append(v(mv_lo(taper(S, cl, 0.0, n))))
    return {r: {k: np.array(v) for k, v in d.items()} for r, d in out.items()}


if __name__ == "__main__":
    res = run()
    print("Random structures, n in {40, 100}, sample covariance singular for T/n < 1.")
    print("Win rate against HRP, and median ratio of variance to HRP's.\n")
    print(f"{'T/n':>6s}" + "".join(f"{h:>22s}" for h in
          ("Ledoit-Wolf min-var", "phi=0.5 taper", "block-diagonal", "NCO", "inverse variance")))
    for r in RATIOS:
        d = res[r]
        cells = []
        for k in ("lw", "taper", "blockdiag", "nco", "ivp"):
            cells.append(f"{np.mean(d[k] < d['hrp']):.0%} / {np.median(d[k]/d['hrp']):.3f}")
        print(f"{r:6.2f}" + "".join(f"{c:>22s}" for c in cells))
    print("\ncell = share of draws beating HRP / median variance ratio to HRP")
    print("\nAbsolute medians")
    print(f"{'T/n':>6s}{'HRP':>10s}{'LW':>10s}{'taper':>10s}{'blockdiag':>11s}{'NCO':>10s}{'inv var':>10s}")
    for r in RATIOS:
        d = res[r]
        print(f"{r:6.2f}" + "".join(f"{np.median(d[k]):10.4f}" for k in
              ("hrp", "lw", "taper", "blockdiag", "nco", "ivp")))
