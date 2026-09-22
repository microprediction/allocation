"""Transport the market's optimum, or estimate the sub-covariance and optimise?

If the index encodes an optimum the market found, a sub-universe portfolio can
be obtained two ways. Estimate the sub-covariance from data and solve, which
is what everyone does and which inherits all the estimation error. Or
transport: calibrate abilities from the market weights and race among the
survivors, which uses NO covariance and so cannot be hurt by a bad one.

Transport does not improve with more data because it never looks at any.
Estimation does. So there should be a crossover in the sample length, and
where it sits is the practical question.

Everything is scored by realized variance under the TRUE covariance.
"""
import sys
import numpy as np
import cvxpy as cp
import winning
from robust import random_structure


def opt(C):
    m = C.shape[0]
    x = cp.Variable(m)
    cp.Problem(cp.Minimize(cp.quad_form(x, cp.psd_wrap(C))),
               [cp.sum(x) == 1, x >= 0]).solve(solver=cp.CLARABEL)
    w = np.maximum(x.value, 0.0)
    return w / w.sum()


def factor_corr(X, k=3):
    """A k-factor correlation from T observations. Far more robustly estimated
    than a full covariance: k(n+1) numbers instead of n(n+1)/2."""
    Xc = X - X.mean(0)
    sd = Xc.std(0, ddof=1); sd[sd == 0] = 1.0
    Z = Xc / sd
    R = (Z.T @ Z) / (len(Z) - 1)
    ev, U = np.linalg.eigh((R + R.T) / 2)
    i = np.argsort(ev)[::-1][:k]
    V = U[:, i] * np.sqrt(np.clip(ev[i], 0.0, None))
    D = np.clip(1.0 - (V ** 2).sum(1), 1e-6, None)
    sc = np.sqrt((V ** 2).sum(1) + D)
    return V / sc[:, None], D / sc ** 2


def run(n=150, keep_frac=0.3, draws=30, seed=8, Ts=(20, 40, 80, 200, 1000)):
    rng = np.random.default_rng(seed)
    keys = (["equal", "rescale", "race"]
            + [f"race+factor T={t}" for t in Ts]
            + [f"estimate T={t}" for t in Ts] + ["oracle"])
    out = {k: [] for k in keys}
    for _ in range(draws):
        Sig, _ = random_structure(rng, n)
        market = opt(Sig)                        # CAPM: the market found this
        m = int(round(keep_frac * n))
        idx = np.sort(rng.choice(n, m, replace=False))
        Sub = Sig[np.ix_(idx, idx)]
        L = np.linalg.cholesky(Sub)
        v = lambda w: float(np.asarray(w, float) @ Sub @ np.asarray(w, float))

        out["equal"].append(v(np.full(m, 1 / m)))
        out["rescale"].append(v(market[idx] / market[idx].sum()))
        a = np.asarray(winning.calibrate_abilities(np.maximum(market, 1e-12)), float)
        p = winning.race_probabilities(a[idx])
        p = np.asarray(p[0] if isinstance(p, tuple) else p, float)
        out["race"].append(v(p / p.sum()))
        out["oracle"].append(v(opt(Sub)))
        for T in Ts:
            X = rng.normal(size=(T, m)) @ L.T
            out[f"estimate T={T}"].append(v(opt(np.cov(X, rowvar=False)
                                               + 1e-8 * np.eye(m))))
            # the middle ground: transport, but let the race see a cheaply
            # estimated factor correlation rather than assume independence
            Vf, Df = factor_corr(X, 3)
            ac = np.asarray(winning.calibrate_abilities(
                np.maximum(market[idx] / market[idx].sum(), 1e-12),
                V=Vf, D=Df), float)
            pc = winning.race_probabilities(ac, V=Vf, D=Df)
            pc = np.asarray(pc[0] if isinstance(pc, tuple) else pc, float)
            out[f"race+factor T={T}"].append(v(pc / pc.sum()))
    return {k: np.asarray(x) for k, x in out.items()}, Ts


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    res, Ts = run(n=n, keep_frac=frac)
    m = int(round(frac * n))
    print(f"universe {n}, sub-universe {m}. True-covariance variance, median of 30.\n")
    base = res["race"]
    print(f"{'method':22s}{'variance':>11s}{'vs race':>10s}{'beats race':>12s}")
    order = (["equal", "rescale", "race"]
             + [x for t in Ts for x in (f"race+factor T={t}", f"estimate T={t}")]
             + ["oracle"])
    for k in order:
        x = res[k]
        note = "" if k == "race" else f"{np.mean(x < base):11.0%}"
        print(f"{k:22s}{np.median(x):11.5f}{np.median(x / base):10.3f}{note}")
