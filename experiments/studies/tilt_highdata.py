"""Third test, aimed at the high-data cells where the effect looked strongest.

Fresh seed, fresh markets, and an extra sample ratio to see whether the trend
continues past parity. Same paired design: tilt(phi) against tilt(0) on the
same seed ensemble, so the race's Monte Carlo error cancels.

If the win rates land inside two standard errors of the first run, the effect
replicates. If they collapse toward 50 percent, the first run was a seed.
"""
import numpy as np
from allocation.thurstone import ThurstonePortfolio
from confound import hrp
from thurstone_rank import SampleCovariance
from tail_test import spec, sample, N, PATHS

RATIOS = (2.0, 5.0)
PHIS = (0.5, 1.0)
OOS = 80_000

FIRST_RUN = {(2.0, 0.5): 0.92, (2.0, 1.0): 0.82}





def race(X, target, phi):
    return np.asarray(
        ThurstonePortfolio(target=target, calib="diagonal", phi=phi, sampler="gaussian",
                           n_paths=PATHS, covariance_estimator=SampleCovariance()
                           ).fit(X).weights_, dtype=float)


def score(panel, w):
    p = panel @ np.asarray(w, dtype=float)
    q = np.quantile(p, 0.05)
    return float(p.var()), float(-p[p <= q].mean())


def run(draws=50, seed=24680):
    rng = np.random.default_rng(seed)
    V = {r: {p: [] for p in PHIS} for r in RATIOS}
    E = {r: {p: [] for p in PHIS} for r in RATIOS}
    for i in range(draws):
        sp = spec(rng)
        panel = sample(rng, sp, OOS)
        for r in RATIOS:
            T = max(int(round(r * N)), 3)
            X = sample(rng, sp, T)
            wh = hrp(np.cov(X, rowvar=False))
            v0, e0 = score(panel, race(X, wh, 0.0))
            for p in PHIS:
                v, e = score(panel, race(X, wh, p))
                V[r][p].append(v / v0); E[r][p].append(e / e0)
        print(f"  ...{i+1}/{draws}", flush=True)
    return V, E


if __name__ == "__main__":
    V, E = run()
    for label, D in (("variance", V), ("expected shortfall", E)):
        print(f"\ntilt(phi) / tilt(0) on {label}\n")
        print(f"{'T/n':>6s}" + "".join(f"{('phi='+str(p)):>26s}" for p in PHIS))
        for r in RATIOS:
            cells = []
            for p in PHIS:
                x = np.array(D[r][p]); w = float(np.mean(x < 1)); n = len(x)
                se = (w * (1 - w) / n) ** 0.5
                cells.append(f"{np.median(x):.4f}  {w:.0%} +/- {se:.0%}")
            print(f"{r:6.2f}" + "".join(f"{c:>26s}" for c in cells))
    print("\nreplication check against the first run (variance win rates)")
    for (r, p), w0 in FIRST_RUN.items():
        x = np.array(V[r][p]); w = float(np.mean(x < 1)); n = len(x)
        se = ((w * (1 - w) / n) + (w0 * (1 - w0) / 40)) ** 0.5
        z = (w - w0) / se if se > 0 else 0.0
        verdict = "replicates" if abs(z) < 2 else "DISAGREES"
        print(f"  T/n={r:4.2f} phi={p}: first {w0:.0%}, now {w:.0%}, z={z:+.2f}  {verdict}")
