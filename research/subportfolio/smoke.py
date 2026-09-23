"""Fast checks. Run this on a new machine before starting anything long.

Each check is a property that has to hold for the study to mean anything, and
each one has caught a real defect in this code at some point. Under a minute.
"""
import sys
import time

import numpy as np

from markets import MidMarket, IndexMarket
from rules import (long_only_min_var, factor_correlation, proportional, race,
                   estimate_and_solve)

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
    idx_mk = IndexMarket(rng, 5000, rank=5)
    check("index parent stationary", idx_mk.premise_residual() < 1e-9,
          f"residual {idx_mk.premise_residual():.1e}")
    check("index parent long only", idx_mk.parent.min() >= 0,
          f"min weight {idx_mk.parent.min():.1e}")

    print("\nan efficient parent is never sparse")
    # No actual investment universe has an efficient portfolio concentrated in
    # a handful of names. A market whose own optimum is sparse is telling you
    # the market is wrong, not that the optimum is. Measured on generic random
    # covariances the long-only minimum-variance portfolio comes back at 2 and
    # 3 effective names out of 300 and 600, with 66 and 56 percent in one name,
    # which is why this study's index market is built around a diffuse parent
    # rather than by solving a covariance drawn first.
    e_parent = 1.0 / float(np.sum(idx_mk.parent ** 2))
    check("index parent is diffuse", e_parent > 0.02 * len(idx_mk.parent),
          f"{e_parent:.0f} effective names of {len(idx_mk.parent)}, "
          f"top weight {idx_mk.parent.max():.3f}")

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
    small = IndexMarket(np.random.default_rng(5), 300, rank=5)
    sub = np.sort(np.random.default_rng(6).choice(300, 40, replace=False))
    pw = proportional(small.parent, sub)
    r, rinfo = race(small.parent, sub)
    l1 = np.abs(r - pw).sum()
    check("race differs from proportional", l1 > 1e-3, f"L1 {l1:.4f}")
    check("race is a portfolio", abs(r.sum() - 1) < 1e-9 and r.min() >= 0)
    check("race reports its calibration", rinfo["converged"],
          f"{rinfo['iterations']} iterations, residual {rinfo['residual']:.1e}")

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
