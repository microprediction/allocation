"""HRP's best case: a market that really is a nested hierarchy.

The paper's claim rests partly on an aggregate over six structural families.
The fair objection is that hierarchical risk parity assumes a hierarchy, so
the test that matters is a market which genuinely has one, strongly, with a
tree the sample can recover. This is that test, and it is generous: the
correlation rises at every level of a clean dyadic nesting, so the structure
HRP looks for is exactly what is there.
"""
import numpy as np
from sklearn.covariance import LedoitWolf
from confound import hrp
from robust import mv_lo, clusters, nco, taper

N = 64


def nested(rng, n=N, base=0.05, add=0.18, levels=(2, 4, 8, 16)):
    """Correlation rises at each level of a dyadic nesting: the cleanest
    possible hierarchy, and the one HRP is built for."""
    C = np.full((n, n), base)
    for lev in levels:
        step = n // lev
        for b in range(lev):
            s = slice(b * step, (b + 1) * step)
            C[s, s] += add
    np.fill_diagonal(C, 1.0)
    ev, V = np.linalg.eigh(C)
    C = V @ np.diag(np.maximum(ev, 1e-8)) @ V.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    vol = np.exp(rng.normal(0, 0.4, n))
    return vol[:, None] * C * vol[None, :]


def run(draws=120, seed=99):
    rng = np.random.default_rng(seed)
    ratios = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
    keys = ("HRP", "inverse variance", "min-var LW", "taper 0.5", "NCO", "true-tree taper")
    out = {r: {k: [] for k in keys} for r in ratios}
    for _ in range(draws):
        Sig = nested(rng)
        L = np.linalg.cholesky(Sig)
        for r in ratios:
            T = max(int(round(r * N)), 4)
            X = rng.normal(size=(T, N)) @ L.T
            S = np.cov(X, rowvar=False)
            v = lambda w: float(np.asarray(w, float) @ Sig @ np.asarray(w, float))
            d = out[r]
            d["HRP"].append(v(hrp(S)))
            iv = 1 / np.diag(S); iv /= iv.sum(); d["inverse variance"].append(v(iv))
            d["min-var LW"].append(v(mv_lo(LedoitWolf(assume_centered=False).fit(X).covariance_)))
            cl = clusters(S, 8)
            d["taper 0.5"].append(v(mv_lo(taper(S, cl, 0.5, N))))
            d["NCO"].append(v(nco(S, cl, N)))
            # the taper handed the TRUE eight-way grouping, not an estimated one
            true_cl = [np.arange(b * (N // 8), (b + 1) * (N // 8)) for b in range(8)]
            d["true-tree taper"].append(v(mv_lo(taper(S, true_cl, 0.5, N))))
    return {r: {k: np.asarray(v) for k, v in dd.items()} for r, dd in out.items()}, ratios, keys


if __name__ == "__main__":
    res, ratios, keys = run()
    others = [k for k in keys if k != "HRP"]
    print(f"A clean dyadic nested hierarchy, n = {N}, 120 draws.")
    print("Share of draws beating HRP / median ratio of variance to HRP's.\n")
    print(f"{'T/n':>6s}" + "".join(f"{k:>22s}" for k in others))
    for r in ratios:
        d = res[r]; h = d["HRP"]
        cells = []
        for k in others:
            x = d[k]
            cells.append(f"{np.mean(x < h):.0%} / {np.median(x / h):.3f}")
        print(f"{r:6.2f}" + "".join(f"{c:>22s}" for c in cells))
    print(f"\n{'T/n':>6s}{'HRP level':>12s}{'best rival':>12s}{'who':>20s}")
    for r in ratios:
        d = res[r]
        best = min(others, key=lambda k: np.median(d[k]))
        print(f"{r:6.2f}{np.median(d['HRP']):12.4f}{np.median(d[best]):12.4f}{best:>20s}")
