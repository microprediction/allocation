"""Does the seriation earn its keep, separately from the split rule?

Hierarchical risk parity has two ideas. Quasi-diagonalization reorders assets
so similar ones sit adjacent, via hierarchical clustering. Recursive bisection
then splits capital by inverse variance down that order. A critique of the
method has to say which half it lands on.

Hold each half fixed and vary the other. On a clean nested hierarchy the true
order is known, so all three orderings can be compared: the one the linkage
recovers, a random one, and the truth.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from confound import cluster_var, min_var
from robust import mv_lo, taper
from nested_best_case import nested, N


def hrp_on_order(S, order):
    """HRP's recursive bisection with inverse-variance splits, down a GIVEN
    order. Passing HRP's own seriation reproduces HRP exactly."""
    n = S.shape[0]
    w = np.ones(n)
    cl = [list(order)]
    while cl:
        nxt = []
        for c in cl:
            if len(c) <= 1:
                continue
            h = len(c) // 2
            l, r = c[:h], c[h:]
            vl, vr = cluster_var(S, l), cluster_var(S, r)
            a = 1.0 - vl / (vl + vr)
            w[l] *= a
            w[r] *= 1.0 - a
            nxt += [l, r]
        cl = nxt
    return w / w.sum()


def linkage_order(S):
    d = np.sqrt(np.diag(S))
    R = np.clip(S / np.outer(d, d), -1, 1)
    D = np.sqrt(np.maximum(0.5 * (1 - R), 0.0))
    np.fill_diagonal(D, 0.0)
    return list(leaves_list(linkage(squareform(D, checks=False), method="single")))


def run(draws=120, seed=606):
    rng = np.random.default_rng(seed)
    ratios = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
    keys = ("HRP, its own seriation", "HRP, random order", "HRP, true order",
            "taper, linkage clusters", "taper, random clusters", "taper, true clusters")
    out = {r: {k: [] for k in keys} for r in ratios}
    true_order = list(range(N))
    true_cl = [np.arange(b * (N // 8), (b + 1) * (N // 8)) for b in range(8)]
    for _ in range(draws):
        Sig = nested(rng)
        L = np.linalg.cholesky(Sig)
        for r in ratios:
            T = max(int(round(r * N)), 4)
            X = rng.normal(size=(T, N)) @ L.T
            S = np.cov(X, rowvar=False)
            v = lambda w: float(np.asarray(w, float) @ Sig @ np.asarray(w, float))
            d = out[r]
            perm = list(rng.permutation(N))
            d["HRP, its own seriation"].append(v(hrp_on_order(S, linkage_order(S))))
            d["HRP, random order"].append(v(hrp_on_order(S, perm)))
            d["HRP, true order"].append(v(hrp_on_order(S, true_order)))
            from robust import clusters as est_clusters
            rnd_cl = [np.asarray(perm[b * (N // 8):(b + 1) * (N // 8)]) for b in range(8)]
            d["taper, linkage clusters"].append(v(mv_lo(taper(S, est_clusters(S, 8), 0.5, N))))
            d["taper, random clusters"].append(v(mv_lo(taper(S, rnd_cl, 0.5, N))))
            d["taper, true clusters"].append(v(mv_lo(taper(S, true_cl, 0.5, N))))
    return {r: {k: np.asarray(v) for k, v in dd.items()} for r, dd in out.items()}, ratios, keys


if __name__ == "__main__":
    res, ratios, keys = run()
    print(f"Clean nested hierarchy, n = {N}, 120 draws. Median true variance.\n")
    print(f"{'T/n':>6s}" + "".join(f"{k[:21]:>24s}" for k in keys))
    for r in ratios:
        print(f"{r:6.2f}" + "".join(f"{np.median(res[r][k]):24.4f}" for k in keys))
    print("\nWhat each half of the idea is worth, as a ratio to the random baseline:\n")
    print(f"{'T/n':>6s}{'seriation, in HRP':>20s}{'grouping, in the taper':>25s}")
    for r in ratios:
        d = res[r]
        a = np.median(d["HRP, its own seriation"] / d["HRP, random order"])
        b = np.median(d["taper, linkage clusters"] / d["taper, random clusters"])
        print(f"{r:6.2f}{a:20.3f}{b:25.3f}")
