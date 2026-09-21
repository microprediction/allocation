"""Does the Thurstone race beat the field where the covariance is singular?

The race inverts nothing: abilities are calibrated so the race under a
reference correlation reproduces a benchmark, then the race is re-run under
the estimated correlation. That is the same structural advantage HRP claims,
reached a different way, so the deeply rank-deficient corner is where it
should show if it shows anywhere.
"""
import numpy as np
from sklearn.covariance import LedoitWolf
from allocation.thurstone import ThurstonePortfolio
from confound import hrp, min_var
from robust import random_structure, mv_lo, clusters, nco, taper


class SampleCovariance:
    """Estimator parity: hand Thurstone the same np.cov every other method
    sees, instead of the package default EWMA, which on a handful of rows is
    a materially different matrix."""
    def __init__(self):
        self.covariance_ = None
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.covariance_ = np.cov(X, rowvar=False)
        return self
    def partial_fit(self, X, y=None):
        return self.fit(X)

RATIOS = (0.1, 0.2, 0.3, 0.5)
METHODS = ("hrp", "ivp", "lw", "taper", "nco",
           "thurstone diag", "thurstone market", "thurstone 3-factor")


def run(draws=80, seed=71):
    rng = np.random.default_rng(seed)
    out = {r: {k: [] for k in METHODS} for r in RATIOS}
    for i in range(draws):
        n = int(rng.choice([40, 100]))
        Sig, fam = random_structure(rng, n)
        L = np.linalg.cholesky(Sig)
        for r in RATIOS:
            T = max(int(round(r * n)), 3)
            X = rng.normal(size=(T, n)) @ L.T
            S = np.cov(X, rowvar=False)
            K = max(2, min(n // 2, int(np.ceil(n / max(T - 1, 1)))))
            cl = clusters(S, min(K, n))
            iv = 1 / np.diag(S); iv /= iv.sum()
            v = lambda w: float(w @ Sig @ w)
            d = out[r]
            d["hrp"].append(v(hrp(S)))
            d["ivp"].append(v(iv))
            d["lw"].append(v(mv_lo(LedoitWolf(assume_centered=False).fit(X).covariance_)))
            d["taper"].append(v(mv_lo(taper(S, cl, 0.5, n))))
            d["nco"].append(v(nco(S, cl, n)))
            for name, kw in (("thurstone diag", dict(calib="diagonal")),
                             ("thurstone market", dict(calib="market")),
                             ("thurstone 3-factor", dict(calib="market", factors=3))):
                try:
                    w = ThurstonePortfolio(n_paths=4096, covariance_estimator=SampleCovariance(), **kw).fit(X).weights_
                    d[name].append(v(np.asarray(w, dtype=float)))
                except Exception:
                    d[name].append(np.nan)
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{draws}", flush=True)
    return {r: {k: np.array(v) for k, v in dd.items()} for r, dd in out.items()}


if __name__ == "__main__":
    res = run()
    print("\nWin rate against HRP / median variance ratio to HRP\n")
    others = [m for m in METHODS if m != "hrp"]
    print(f"{'T/n':>6s}" + "".join(f"{m:>22s}" for m in others))
    for r in RATIOS:
        d = res[r]; h = d["hrp"]
        cells = []
        for m in others:
            x = d[m]; ok = np.isfinite(x)
            cells.append(f"{np.mean(x[ok] < h[ok]):.0%} / {np.median(x[ok]/h[ok]):.3f}")
        print(f"{r:6.2f}" + "".join(f"{c:>22s}" for c in cells))
    print("\nBest method by median variance ratio to HRP, per ratio")
    for r in RATIOS:
        d = res[r]; h = d["hrp"]
        sc = {m: np.median(d[m][np.isfinite(d[m])] / h[np.isfinite(d[m])]) for m in others}
        best = sorted(sc.items(), key=lambda kv: kv[1])[:3]
        print(f"  T/n={r:4.2f}: " + ",  ".join(f"{k} {v:.3f}" for k, v in best))
