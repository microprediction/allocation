"""Confirm or kill the one positive signal, with enough draws to tell.

On the crash market the Gaussian tilt of HRP looked 2 to 6 percent better at
T/n >= 0.3, winning 56 to 68 percent of 25 draws. At 25 draws a 64 percent
win rate is about 1.5 standard errors from a coin, so that is suggestive and
nothing more.

This run pairs under common random numbers, tilt(phi) against tilt(0) on the
same seed ensemble, which removes the race's Monte Carlo noise, and uses
enough draws that a real 60 percent win rate separates from 50.
"""
import numpy as np
from allocation.thurstone import ThurstonePortfolio
from confound import hrp
from thurstone_rank import SampleCovariance
from tail_test import spec, sample, N, PATHS

RATIOS = (0.3, 1.0)
PHIS = (0.5, 1.0)
OOS = 80_000


def race(X, target, phi, sampler="gaussian"):
    return np.asarray(
        ThurstonePortfolio(target=target, calib="diagonal", phi=phi, sampler=sampler,
                           nu=5.0, n_paths=PATHS,
                           covariance_estimator=SampleCovariance()).fit(X).weights_,
        dtype=float)


def score(panel, w):
    p = panel @ np.asarray(w, dtype=float)
    q = np.quantile(p, 0.05)
    return float(p.var()), float(-p[p <= q].mean())


def run(draws=60, seed=13579):
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
        print(f"\ntilt(phi) / tilt(0) on {label}: median, win rate, and its standard error\n")
        print(f"{'T/n':>6s}" + "".join(f"{('phi='+str(p)):>28s}" for p in PHIS))
        for r in RATIOS:
            cells = []
            for p in PHIS:
                x = np.array(D[r][p]); w = float(np.mean(x < 1)); n = len(x)
                se = (w * (1 - w) / n) ** 0.5
                cells.append(f"{np.median(x):.4f}  {w:.0%} +/- {se:.0%}")
            print(f"{r:6.2f}" + "".join(f"{c:>28s}" for c in cells))
