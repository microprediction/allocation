"""Fast checks. Run this on a new machine before starting anything long.

Each check is a property that has to hold for the study to mean anything, and
each one has caught a real defect in this code at some point. Under a minute.
"""
import sys
import time

import numpy as np

from markets import (MidMarket, IndexMarket, effective_fraction,
                     REAL_EFFN_FRACTION)
from rules import (long_only_min_var, long_only_max_sharpe, factor_correlation,
                   proportional, race, flattened, black_litterman,
                   estimate_and_solve)

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)

    print("\npremise: the parent is the tangency portfolio of (Sigma, m)")
    mid = MidMarket(rng, 120, solver=long_only_min_var)
    check("mid parent stationary", mid.premise_residual() < 1e-6,
          f"residual {mid.premise_residual():.1e}")
    idx_mk = IndexMarket(rng, 5000)
    check("index parent stationary", idx_mk.premise_residual() < 1e-9,
          f"residual {idx_mk.premise_residual():.1e}")
    check("index parent long only", idx_mk.parent.min() >= 0,
          f"min weight {idx_mk.parent.min():.1e}")

    print("\nthe parent is an index, not a corner solution")
    lo, hi = REAL_EFFN_FRACTION
    for label, mk, n in (("mid", mid, 120), ("index", idx_mk, 5000)):
        w = mk.parent
        check(f"{label} parent holds every name", w.min() > 0,
              f"min weight {w.min():.1e}, {100 * np.mean(w > 1e-8):.0f}% held")
    # The old mid market solved for a long-only minimum-variance parent, which
    # is a corner: 25 names of 400, and one draw held 6. The race floors the
    # rest at 1e-12, so their abilities were one-sided bounds and not
    # information, and a random sub-universe held 0 to 4 of the 60. Every rule
    # was then splitting the same near point mass, which is why race came out
    # at 0.999 of proportional. Concentration is a premise here, so it is
    # checked rather than assumed.
    fr = [effective_fraction(MidMarket(np.random.default_rng([12, g]), 400).parent)
          for g in range(6)]
    check("mid parent no more concentrated than the real index",
          min(fr) >= lo, f"effective fraction {min(fr):.3f} to {max(fr):.3f}, "
                         f"real index {lo:.2f} to {hi:.2f}")
    check("index parent no more concentrated than the real index",
          effective_fraction(idx_mk.parent) * 5000 > 40,
          f"{1 / np.sum(idx_mk.parent ** 2):.0f} effective names of 5000")

    print("\nmarket: the blocks are usable covariances")
    idx = np.sort(np.random.default_rng(1).choice(5000, 200, replace=False))
    S = idx_mk.block(idx)
    ev = np.linalg.eigvalsh(S)
    sd = np.sqrt(np.diag(S))
    R = S / np.outer(sd, sd)
    off = R[~np.eye(len(idx), dtype=bool)]
    check("index block positive definite", ev.min() > 0, f"min eig {ev.min():.2e}")
    check("index correlations plausible", 0.1 < off.mean() < 0.5,
          f"mean {off.mean():.2f}, range {off.min():.2f} to {off.max():.2f}")
    evc = np.linalg.eigvalsh(R)[::-1] / len(idx)
    check("index market is not low rank", evc[0] < 0.45 and evc[2:].sum() > 0.45,
          f"PC1 {evc[0]:.0%}, PC2 {evc[1]:.1%}, everything past PC2 {evc[2:].sum():.0%}"
          " -- a k-factor estimate cannot capture this")

    print("\npanel: the simulated data really comes from that covariance")
    X = idx_mk.panel(np.random.default_rng(2), 40000)[:, idx]
    err = np.abs(np.cov(X, rowvar=False) - S).max() / np.abs(S).max()
    check("index panel matches its block", err < 0.05, f"max rel err {err:.3f}")

    print("\ncorrelation factors: unit diagonal, no volatility smuggled in")
    sd_true = np.exp(np.random.default_rng(3).normal(0, 0.7, 200))
    Y = np.random.default_rng(4).normal(size=(500, 200)) * sd_true
    _, V, D = factor_correlation(Y, 3)
    err = np.abs((V ** 2).sum(1) + D - 1).max()
    check("V V' + D has unit diagonal", err < 1e-12, f"max departure {err:.1e}")

    print("\nthe race is a restriction, not a relabelling")
    small = IndexMarket(np.random.default_rng(5), 300)
    sub = np.sort(np.random.default_rng(6).choice(300, 40, replace=False))
    pw = proportional(small.parent, sub)
    r = race(small.parent, sub)
    l1 = np.abs(r - pw).sum()
    check("race differs from proportional", l1 > 1e-3, f"L1 {l1:.4f}")
    check("race is a portfolio", abs(r.sum() - 1) < 1e-9 and r.min() >= 0)

    print("\nthe independent race is a power transform, so it needs a better null")
    fl = flattened(small.parent, sub)
    check("race is close to flattened proportional",
          np.abs(r - fl).sum() < 0.2 * l1,
          f"L1(race, flattened) {np.abs(r - fl).sum():.4f} vs "
          f"L1(race, proportional) {l1:.4f} "
          "-- this is why proportional is the wrong null")

    print("\ncalibrating on the restricted field would return the input")
    a = np.asarray(__import__("winning").calibrate_abilities(
        np.maximum(pw, 1e-12), target_floor=1e-12), float)
    back = np.asarray(__import__("winning").race_probabilities(a), float)
    back = back / back.sum()
    check("round trip on the sub-field is the identity",
          np.abs(back - pw).sum() < 1e-6,
          f"L1 {np.abs(back - pw).sum():.2e} (this is the trap, not the method)")

    print("\nBlack-Litterman on the true covariance is the oracle")
    Sg = mid.Sigma
    sdv = np.sqrt(np.diag(Sg))
    Cc = Sg / np.outer(sdv, sdv)
    evc, Qc = np.linalg.eigh((Cc + Cc.T) / 2)
    Vx = Qc * np.sqrt(np.maximum(evc, 0))
    Dx = np.maximum(1.0 - (Vx ** 2).sum(1), 0)
    mid_sub = np.sort(np.random.default_rng(11).choice(mid.n, 40, replace=False))
    bl = black_litterman(mid.parent, mid_sub, sdv, Vx, Dx)
    orc = long_only_max_sharpe(mid.block(mid_sub), mid.m[mid_sub])
    check("BL with the true covariance reproduces the oracle",
          np.abs(bl - orc).sum() < 1e-6,
          f"L1 {np.abs(bl - orc).sum():.2e} (so its gap from the oracle is "
          "entirely the covariance estimate)")

    print("\nrestriction is not renormalisation")
    Ssm = small.block(sub)
    msm = small.m[sub]
    w = long_only_max_sharpe(Ssm, msm)
    gap = np.abs(w - pw).sum()
    check("the sub-universe optimum differs from proportional", gap > 0.05,
          f"L1 {gap:.3f}: the departed names' weight does not just rescale")

    print("\nthe oracle really is optimal")
    sh = lambda x: float(x @ msm) / float(np.sqrt(x @ Ssm @ x))
    others = [sh(pw), sh(r), sh(fl), sh(np.full(40, 1 / 40))]
    check("oracle beats every candidate", all(sh(w) >= o - 1e-10 for o in others),
          f"oracle sharpe {sh(w):.4f} vs best other {max(others):.4f}")

    print(f"\n{len(FAILS)} failures, {time.time() - t0:.0f}s")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
