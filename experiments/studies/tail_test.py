"""The tail test, on a market where the tail is not a function of variance.

Calm regime: one random dependence structure. Crash regime, firing 8 percent
of the time: an INDEPENDENT structure over a different labelling of the
assets, three times the volatility, negative mean. A portfolio can therefore
look diversified under the sample covariance and be concentrated in whatever
co-moves during a crash. Validated: shortfall over standard deviation varies
by 17 percent across portfolios here against 1 percent under multivariate t.

Every portfolio is scored on the same out-of-sample panel by variance and by
expected shortfall at 5 percent. The question is whether the Student-t race
buys anything in the second column that it does not buy in the first.
"""
import numpy as np
from sklearn.covariance import LedoitWolf
from allocation.thurstone import ThurstonePortfolio
from confound import hrp
from robust import random_structure, mv_lo, clusters, taper
from thurstone_rank import SampleCovariance

N, PATHS, OOS, P_CRASH = 40, 16384, 120_000, 0.08
RATIOS = (0.1, 0.3, 1.0)


def spec(rng):
    Sc, _ = random_structure(rng, N)
    So, _ = random_structure(rng, N)
    perm = rng.permutation(N)
    Sx = So[np.ix_(perm, perm)]
    d = np.sqrt(np.diag(Sc)); dc = np.sqrt(np.diag(Sx))
    Sx = (3.0 * d / dc)[:, None] * Sx * (3.0 * d / dc)[None, :]
    return Sc, Sx, -2.0 * d


def sample(rng, sp, rows):
    Sc, Sx, mu = sp
    Lc = np.linalg.cholesky(Sc); Lx = np.linalg.cholesky(Sx)
    crash = rng.random(rows) < P_CRASH
    Z = rng.normal(size=(rows, N))
    out = Z @ Lc.T
    if crash.any():
        out[crash] = Z[crash] @ Lx.T + mu
    return out


def race(X, target, phi, sampler):
    return np.asarray(
        ThurstonePortfolio(target=target, calib="diagonal", phi=phi, sampler=sampler,
                           nu=5.0, n_paths=PATHS,
                           covariance_estimator=SampleCovariance()).fit(X).weights_,
        dtype=float)


def run(draws=25, seed=6174):
    rng = np.random.default_rng(seed)
    keys = ["hrp", "ivp", "lw", "taper",
            "gauss tilt 0.5", "gauss tilt 1.0", "t tilt 0.5", "t tilt 1.0"]
    var = {r: {k: [] for k in keys} for r in RATIOS}
    es = {r: {k: [] for k in keys} for r in RATIOS}
    for i in range(draws):
        sp = spec(rng)
        panel = sample(rng, sp, OOS)
        for r in RATIOS:
            T = max(int(round(r * N)), 3)
            X = sample(rng, sp, T)
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
    print("\ndoes the tail column rank differently from the variance column?")
    for r in RATIOS:
        dv, de = var[r], es[r]
        rv = sorted(others, key=lambda k: np.median(np.array(dv[k]) / np.array(dv["hrp"])))
        re = sorted(others, key=lambda k: np.median(np.array(de[k]) / np.array(de["hrp"])))
        print(f"  T/n={r:4.2f}  variance: {rv[0]}, {rv[1]}   shortfall: {re[0]}, {re[1]}"
              f"   {'SAME' if rv[:2] == re[:2] else 'DIFFERENT'}")
