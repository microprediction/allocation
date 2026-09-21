"""The tilt measured against itself, so the Monte Carlo error cancels.

The race is simulated, so its weights carry sampling noise that scales as
1/sqrt(paths): at 4096 paths and 100 assets the L1 noise is 0.118, which
swamps a tilt effect of a few percent. But the implementation transports a
FIXED seed ensemble across phi, so comparing the race at phi against the
same race at phi = 0 is a paired comparison under common random numbers and
most of that noise cancels.

So the estimand is variance(tilt at phi) / variance(tilt at 0), not a
comparison against exact HRP. The pedestal, how far the phi = 0 race sits
from exact HRP, is reported separately rather than mixed into the effect.
"""
import numpy as np
from allocation.thurstone import ThurstonePortfolio
from confound import hrp
from robust import random_structure
from thurstone_rank import SampleCovariance

RATIOS = (0.1, 0.3)
PHIS = (0.25, 0.5, 0.75, 1.0)
PATHS = 16384
N = 40


def race(X, target, phi):
    return np.asarray(
        ThurstonePortfolio(target=target, calib="diagonal", phi=phi, n_paths=PATHS,
                           covariance_estimator=SampleCovariance()).fit(X).weights_,
        dtype=float)


def run(draws=30, seed=1234):
    rng = np.random.default_rng(seed)
    out = {r: {p: [] for p in PHIS} for r in RATIOS}
    ped, base = [], {r: [] for r in RATIOS}
    for i in range(draws):
        Sig, fam = random_structure(rng, N)
        L = np.linalg.cholesky(Sig)
        for r in RATIOS:
            T = max(int(round(r * N)), 3)
            X = rng.normal(size=(T, N)) @ L.T
            S = np.cov(X, rowvar=False)
            wh = hrp(S)
            v = lambda w: float(w @ Sig @ w)
            w0 = race(X, wh, 0.0)
            ped.append(float(np.abs(w0 - wh).sum()))
            base[r].append(v(w0) / v(wh))
            for p in PHIS:
                out[r][p].append(v(race(X, wh, p)) / v(w0))
        print(f"  ...{i+1}/{draws}", flush=True)
    return out, np.array(ped), {r: np.array(v) for r, v in base.items()}


if __name__ == "__main__":
    out, ped, base = run()
    print(f"\nMonte Carlo pedestal at {PATHS} paths, n = {N}:")
    print(f"  median L1 gap between the phi=0 race and exact HRP {np.median(ped):.4f}")
    for r in RATIOS:
        print(f"  T/n={r}: variance of the phi=0 race / exact HRP  {np.median(base[r]):.4f}")
    print("\nTilt effect, paired under common random numbers.")
    print("variance(tilt at phi) / variance(tilt at 0), median and share below 1\n")
    print(f"{'T/n':>6s}" + "".join(f"{('phi='+str(p)):>20s}" for p in PHIS))
    for r in RATIOS:
        cells = []
        for p in PHIS:
            x = np.array(out[r][p])
            cells.append(f"{np.median(x):.4f} ({np.mean(x < 1):.0%})")
        print(f"{r:6.2f}" + "".join(f"{c:>20s}" for c in cells))
