"""Sharded runner. One draw is one independent unit of work.

Each draw builds a market, takes a random sub-universe, applies every rule,
and scores the Sharpe ratio against the TRUE sub-covariance and the market's
own implied returns. There is no backtest and no estimation in the scoring, so
a difference between rules is a difference between rules.

Sharding is by global draw index, and every draw seeds itself from
(seed, draw_index). A draw therefore produces the same numbers whether it ran
alone, in a shard of four or in a shard of sixty-four, which is what makes it
safe to spread this over a big machine and merge afterwards.

    python run.py --scale mid   --n 400  --m 60  --draws 25
    python run.py --scale index --n 5000 --m 200 --draws 32 --k 2 \
                  --shard 3 --shards 16

Cost is dominated by the k-factor calibration inside winning, which is
exponential in k and linear in n (winning issue #156). See README.md for the
measured runtime model before choosing sizes.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import winning

from markets import MARKETS
from rules import (long_only_min_var, long_only_max_sharpe, factor_correlation,
                   factor_covariance, proportional, race, flattened,
                   black_litterman, estimate_and_solve)

PREMISE_TOL = 1e-9      # the premise holds by construction; this catches bugs


def one_draw(args, g):
    """Run global draw index g. Returns a dict of method -> variance."""
    rng = np.random.default_rng([args.seed, g])
    Market = MARKETS[args.scale]
    mk = Market(rng, args.n, solver=long_only_min_var)

    resid = mk.premise_residual()
    if resid > PREMISE_TOL:
        return {"draw": g, "skipped": "premise", "premise_residual": resid}

    idx = np.sort(rng.choice(args.n, args.m, replace=False))
    Sub = mk.block(idx)
    m_S = mk.m[idx]
    ev = np.linalg.eigvalsh(Sub)

    def score(w):
        w = np.asarray(w, float)
        return float(w @ m_S) / float(np.sqrt(w @ Sub @ w))

    row = {"draw": g, "family": mk.family, "objective": "sharpe",
           "premise_residual": resid,
           "min_eig": float(ev.min()), "parent_effective_n":
           float(1.0 / np.sum(mk.parent ** 2))}
    row["equal weight"] = score(np.full(args.m, 1.0 / args.m))
    row["proportional"] = score(proportional(mk.parent, idx))
    row["flattened"] = score(flattened(mk.parent, idx))
    row["race"] = score(race(mk.parent, idx))
    row["oracle"] = score(long_only_max_sharpe(Sub, m_S))

    # Two T grids, because the two rules cost three orders of magnitude apart.
    # race+factor calibrates under a k-factor correlation, which is 68 core-
    # seconds at k=3 and n=400, so it stays on a coarse grid. estimate+solve is
    # a cvxpy solve at 0.12s, and `race` never sees the panel at all and is
    # therefore constant in T. The question the fine grid answers -- how little
    # data it takes before estimating beats believing the parent -- is a flat
    # line against a cheap curve, so resolving it well costs nearly nothing.
    # One panel per T, shared by both rules, so the comparison stays paired.
    est_Ts = args.Ts_est or args.Ts
    for T in sorted(set(args.Ts) | set(est_Ts)):
        X = mk.panel(rng, T)
        if T in set(args.Ts):
            _, V, D = factor_correlation(X, args.k)
            row[f"race+factor T={T}"] = score(race(mk.parent, idx, V=V, D=D))
        if T in set(est_Ts):
            row[f"estimate+solve T={T}"] = score(estimate_and_solve(X, idx))
            sd, Vc, Dc = factor_covariance(X, args.k)
            row[f"black-litterman T={T}"] = score(
                black_litterman(mk.parent, idx, sd, Vc, Dc))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale", choices=sorted(MARKETS), default="mid")
    p.add_argument("--n", type=int, default=400, help="parent universe size")
    p.add_argument("--m", type=int, default=60, help="sub-universe size")
    p.add_argument("--draws", type=int, default=25)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--k", type=int, default=3, help="factors in the estimated correlation")
    p.add_argument("--Ts", type=int, nargs="+", default=[20, 40, 100],
                   help="panel lengths for race+factor (the expensive rule)")
    p.add_argument("--Ts-est", type=int, nargs="+", default=None,
                   help="panel lengths for estimate+solve; defaults to --Ts. "
                        "Cheap, so use a fine grid to locate the crossover.")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default=None)
    a = p.parse_args()

    if not 0 <= a.shard < a.shards:
        p.error("--shard must be in [0, --shards)")
    tag = a.tag or f"{a.scale}-n{a.n}-m{a.m}-k{a.k}-s{a.seed}"
    mine = [g for g in range(a.draws) if g % a.shards == a.shard]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    dest = Path(a.out) / f"{tag}-shard{a.shard}of{a.shards}.json"

    rows, t0 = [], time.time()
    env = {"host": platform.node(), "python": platform.python_version(),
           "numpy": np.__version__,
           "winning": getattr(winning, "__version__", "unknown"),
           "winning_path": os.path.dirname(winning.__file__),
           "threads": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                        # fastrace links rayon and reads only this one; none
                        # of the BLAS variables above reach the race kernel.
                        "RAYON_NUM_THREADS")}}

    def checkpoint():
        """Write the shard atomically, so a stopped run keeps what it had.

        Written after every draw. A plain write truncates first, so a process
        killed mid-write would lose every draw before it as well; the rename is
        atomic on POSIX, and the file is always either the previous complete
        snapshot or the new one.
        """
        blob = dict(config=vars(a), rows=rows,
                    elapsed_seconds=time.time() - t0, **env)
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(blob, indent=1, default=str))
        os.replace(tmp, dest)

    for j, g in enumerate(mine):
        td = time.time()
        rows.append(one_draw(a, g))
        checkpoint()
        print(f"  shard {a.shard}/{a.shards}  draw {g}  "
              f"({j + 1}/{len(mine)})  {time.time() - td:.1f}s", flush=True)

    checkpoint()
    print(f"\nwrote {dest}  ({len(rows)} draws, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
