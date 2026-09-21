"""Score the tilt on the objective it was built for.

The Thurstone race with a Student-t sampler has fat marginal tails AND tail
dependence, so its winning probabilities are a genuine tail statistic rather
than a function of the correlation alone. Judging that on variance is the
scoring mismatch. Here the market itself is multivariate t, and every
portfolio is scored twice on the SAME out-of-sample panel: by variance and by
expected shortfall at 5 percent.

If the tilt earns its keep anywhere it is in the gap between those two
columns.
"""
import numpy as np
from sklearn.covariance import LedoitWolf
from allocation.thurstone import ThurstonePortfolio
from confound import hrp
from robust import random_structure, mv_lo, clusters, taper
from thurstone_rank import SampleCovariance

N, NU, PATHS, OOS = 40, 5.0, 16384, 120_000
RATIOS = (0.1, 0.3)


def draw_t(rng, Sig, rows):
    """Multivariate t with nu degrees of freedom and covariance exactly Sig."""
    L = np.linalg.cholesky(Sig * (NU - 2.0) / NU)
    Z = rng.normal(size=(rows, N)) @ L.T
    W = rng.chisquare(NU, size=(rows, 1)) / NU
    return Z / np.sqrt(W)


def race(X, target, phi, sampler):
    return np.asarray(
        ThurstonePortfolio(target=target, calib="diagonal", phi=phi, sampler=sampler,
                           nu=NU, n_paths=PATHS,
                           covariance_estimator=SampleCovariance()).fit(X).weights_,
        dtype=float)


def run(draws=25, seed=2718):
    rng = np.random.default_rng(seed)
    keys = ["hrp", "ivp", "lw", "taper",
            "gauss tilt 0.5", "gauss tilt 1.0", "t tilt 0.5", "t tilt 1.0"]
    var = {r: {k: [] for k in keys} for r in RATIOS}
    es = {r: {k: [] for k in keys} for r in RATIOS}
    for i in range(draws):
        Sig, fam = random_structure(rng, N)
        panel = draw_t(rng, Sig, OOS)
        for r in RATIOS:
            T = max(int(round(r * N)), 3)
            X = draw_t(rng, Sig, T)
            S = np.cov(X, rowvar=False)
            K = max(2, min(N // 2, int(np.ceil(N / max(T - 1, 1)))))
            cl = clusters(S, min(K, N))
            iv = 1 / np.diag(S); iv /= iv.sum()
            wh = hrp(S)
            ws = {"hrp": wh, "ivp": iv,
                  "lw": mv_lo(LedoitWolf(assume_centered=False).fit(X).covariance_),
                  "taper": mv_lo(taper(S, cl, 0.5, N)),
                  "gauss tilt 0.5": race(X, wh, 0.5, "gaussian"),
                  "gauss tilt 1.0": race(X, wh, 1.0, "gaussian"),
                  "t tilt 0.5": race(X, wh, 0.5, "student_t"),
                  "t tilt 1.0": race(X, wh, 1.0, "student_t")}
            for k, w in ws.items():
                p = panel @ np.asarray(w, dtype=float)
                var[r][k].append(float(p.var()))
                q = np.quantile(p, 0.05)
                es[r][k].append(float(-p[p <= q].mean()))
        print(f"  ...{i+1}/{draws}", flush=True)
    return var, es, keys


if __name__ == "__main__":
    var, es, keys = run()
    others = [k for k in keys if k != "hrp"]
    for label, D in (("variance", var), ("expected shortfall at 5%", es)):
        print(f"\nratio to HRP on {label}: median (share better than HRP)\n")
        print(f"{'T/n':>6s}" + "".join(f"{k:>20s}" for k in others))
        for r in RATIOS:
            d = D[r]; h = np.array(d["hrp"])
            cells = []
            for k in others:
                x = np.array(d[k])
                cells.append(f"{np.median(x/h):.4f} ({np.mean(x < h):.0%})")
            print(f"{r:6.2f}" + "".join(f"{c:>20s}" for c in cells))
