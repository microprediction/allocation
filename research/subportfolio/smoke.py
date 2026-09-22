"""Fast checks. Run this on a new machine before starting anything long.

Each check is a property that has to hold for the study to mean anything, and
each one has caught a real defect in this code at some point. Under a minute.
"""
import sys
import time

import numpy as np

from markets import (MidMarket, IndexMarket, effective_fraction,
                     REAL_EFFN_FRACTION)
from rules import (long_only_min_var, factor_correlation, proportional, race,
                   flattened, estimate_and_solve)

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)

    print("\npremise: the parent is the exact optimum of the true covariance")
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

    print("\nthe oracle really is optimal")
    Ssm = small.block(sub)
    w = long_only_min_var(Ssm)
    v = lambda x: float(x @ Ssm @ x)
    others = [v(pw), v(r), v(np.full(40, 1 / 40))]
    check("oracle beats every candidate", all(v(w) <= o + 1e-10 for o in others),
          f"oracle {v(w):.5f} vs best other {min(others):.5f}")

    print(f"\n{len(FAILS)} failures, {time.time() - t0:.0f}s")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
