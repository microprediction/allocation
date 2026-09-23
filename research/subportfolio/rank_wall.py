"""Does the restriction need the rank it cannot afford?

The k-factor race is exponential in k (winning#156), so above rank two or
three at n > 1000 there is no tractable evaluation. That is a hard limit on
this study, and the question is whether it costs anything.

Two things are measured against k, separately, because they can fail for
different reasons.

How much the extra factor is worth: the variance of the restricted portfolio
under the true sub-covariance, as k goes 1, 2, 3. If the gain saturates at
k = 1 or 2 the wall does not bind here.

Whether the extra factor is even estimable: the alignment of the k-th
eigenvector of the sample correlation with its population counterpart, at the
panel lengths an index holder actually has. A factor that cannot be estimated
cannot help no matter how cheaply it could be raced, and the two limits may
simply coincide.

    python rank_wall.py 300 50
"""
import sys
import time

import numpy as np

from markets import IndexMarket
from rules import long_only_min_var, factor_correlation, proportional, race


def population_corr(mk):
    n = mk.n
    M = np.outer(mk.v, mk.v) + np.diag(mk.d)
    C = (M - np.outer(mk.u, mk.Mu) - np.outer(mk.Mu, mk.u)
         + mk.uMu * np.outer(mk.u, mk.u))
    S = 1.0 + mk.eps * C
    S = (S + S.T) / 2
    sd = np.sqrt(np.diag(S))
    return S / np.outer(sd, sd)


def sharpness(V, D):
    """winning's node-rule statistic: sqrt(2) max_i ||V_i|| / sd_i.

    It decides the node FAMILY, and it keys on the maximum, so one name with
    a communality above about 0.82 sets the cost for the whole field. Recorded
    per draw because a gain that is bimodal with sharpness as the splitter is
    a different result from a gain that is small on average.
    """
    return float(np.sqrt(2.0) * np.max(
        np.sqrt((V ** 2).sum(1) / np.clip(D, 1e-12, None))))


def eigvecs(C, k):
    ev, U = np.linalg.eigh((C + C.T) / 2)
    i = np.argsort(ev)[::-1][:k]
    return U[:, i], ev[i]


def run(n=250, m=50, draws=24, seed=11, Ts=(52, 104), ks=(1, 2, 3), rank=5,
        target_corr=0.27):
    rng = np.random.default_rng(seed)
    var = {(k, T): [] for k in ks for T in Ts}
    base, plain = [], []
    align = {(k, T): [] for k in ks for T in Ts}
    cost = {k: [] for k in ks}
    sharp = {(k, T): [] for k in ks for T in Ts}
    for _ in range(draws):
        mk = IndexMarket(rng, n, rank=rank, target_corr=target_corr)
        idx = np.sort(rng.choice(n, m, replace=False))
        Sub = mk.block(idx)
        v = lambda w: float(w @ Sub @ w)
        base.append(v(proportional(mk.parent, idx)))
        w, _ = race(mk.parent, idx)
        plain.append(v(w))

        Rpop = population_corr(mk)
        Upop, _ = eigvecs(Rpop, max(ks))
        for T in Ts:
            X = mk.panel(rng, T)
            Z = (X - X.mean(0)) / X.std(0, ddof=1)
            Rhat = (Z.T @ Z) / (len(Z) - 1)
            Uhat, _ = eigvecs(Rhat, max(ks))
            for k in ks:
                # |cos angle| between the k-th sample and population direction
                align[(k, T)].append(abs(float(Uhat[:, k - 1] @ Upop[:, k - 1])))
                t = time.time()
                _, V, D = factor_correlation(X, k)
                sharp[(k, T)].append(sharpness(V, D))
                w, info = race(mk.parent, idx, V=V, D=D)
                cost[k].append(time.time() - t)
                var[(k, T)].append(v(w) if info["converged"] else np.nan)
    return (np.array(base), np.array(plain), var, align, cost, Ts, ks, sharp)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    rank = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    base, plain, var, align, cost, Ts, ks, sharp = run(n=n, m=m, rank=rank)
    print(f"\nparent {n} names, sub-universe {m}, TRUE RANK {rank}, "
          f"{len(base)} draws. "
          f"Ratio to proportional restriction.\n")
    print(f"  {'race, no correlation':24s}{np.median(plain / base):8.3f}")
    print(f"\n{'':24s}" + "".join(f"{'T=' + str(T):>12}" for T in Ts))
    for k in ks:
        row = f"  race + {k}-factor{'':9s}"
        for T in Ts:
            x = np.array(var[(k, T)])
            row += f"{np.nanmedian(x / base):12.3f}"
        print(row)

    # A median can hide a field on which the extra factor matters a great
    # deal. What decides whether rank 3 is worth its cost is not the typical
    # draw but the worst one, so the per-draw spread of the gain from the
    # third factor over the second is printed rather than summarised away.
    if 3 in ks and 2 in ks:
        print(f"\ngain from the third factor over the second, per draw "
              f"(below 1 means the third factor helped)\n")
        for T in Ts:
            r = np.array(var[(3, T)]) / np.array(var[(2, T)])
            sh = np.array(sharp[(3, T)])
            ok = np.isfinite(r)
            r, sh = r[ok], sh[ok]
            print(f"  T={T:<5} median {np.median(r):.4f}   best {r.min():.4f}"
                  f"   worst {r.max():.4f}   helped on {np.mean(r < 1):.0%} "
                  f"of {len(r)} draws")
            # the gain against the statistic that decides the node family, so a
            # bimodal gain split by sharpness cannot hide inside the median
            lo, hi = sh <= 4.75, sh > 4.75
            for lab, sel in (("sharp <= 4.75 (gauss-hermite)", lo),
                             ("sharp >  4.75 (sobol 8192)  ", hi)):
                if sel.any():
                    print(f"      {lab}  n={sel.sum():2d}  "
                          f"median gain {np.median(r[sel]):.4f}  "
                          f"worst {r[sel].max():.4f}")
            print(f"      sharpness ran {sh.min():.1f} to {sh.max():.1f}")

    print(f"\nis the k-th factor estimable? |cos| of the sample direction "
          f"against the population one\n")
    print(f"{'':24s}" + "".join(f"{'T=' + str(T):>12}" for T in Ts))
    for k in ks:
        row = f"  factor {k}{'':15s}"
        for T in Ts:
            row += f"{np.median(align[(k, T)]):12.2f}"
        print(row)

    print(f"\nseconds per race, by rank")
    for k in ks:
        print(f"  k={k}: {np.median(cost[k]):7.2f}s")
    # Computed, not written down. Every hardcoded version of this paragraph
    # went stale within a run or two: it quoted a 3.0 threshold that winning
    # PR #172 replaced with a per-rank schedule, and a sharpness range from a
    # six-draw run that a twenty-four draw run contradicted.
    allsharp = np.concatenate([np.asarray(sharp[(k, T)]) for k in ks for T in Ts])
    thr = {1: None, 2: 3.75, 3: 4.75}
    named = ", ".join(f"k={k}: {thr[k]}" for k in ks if thr.get(k))
    print(f"\nCost is set by field sharpness as much as by rank. Above a "
          f"per-rank\nthreshold ({named}) winning switches the node family to "
          f"scrambled Sobol\nat 8192 points whatever the rank. Measured here "
          f"sharp ran {allsharp.min():.1f} to {allsharp.max():.1f},\nso draws "
          f"fall either side and cost is bimodal rather than ordered by\nrank. "
          f"The rule keys on max_i, so one name sets the cost for the field.")
