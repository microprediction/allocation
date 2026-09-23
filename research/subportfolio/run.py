"""Sharded runner. One draw is one independent unit of work.

Each draw builds a market, takes a random sub-universe, applies every rule,
and scores realized variance against the TRUE sub-covariance. There is no
backtest and no estimation in the scoring, so a difference between rules is a
difference between rules.

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
from rules import (long_only_min_var, factor_correlation, proportional, race,
                   estimate_and_solve)

PREMISE_TOL = 1e-5      # mid draws land near 3e-07, index near 1e-14


def one_draw(args, g):
    """Run global draw index g. Returns a dict of method -> variance."""
    rng = np.random.default_rng([args.seed, g])
    Market = MARKETS[args.scale]
    kw = {} if args.scale == "mid" else {"rank": args.rank,
                                         "target_corr": args.target_corr}
    mk = Market(rng, args.n, solver=long_only_min_var, **kw)

    resid = mk.premise_residual()
    if resid > PREMISE_TOL:
        return {"draw": g, "skipped": "premise", "premise_residual": resid}

    idx = np.sort(rng.choice(args.n, args.m, replace=False))
    Sub = mk.block(idx)
    ev = np.linalg.eigvalsh(Sub)
    var = lambda w: float(np.asarray(w, float) @ Sub @ np.asarray(w, float))

    row = {"draw": g, "family": mk.family, "premise_residual": resid,
           "min_eig": float(ev.min()), "parent_effective_n":
           float(1.0 / np.sum(mk.parent ** 2))}
    row["equal weight"] = var(np.full(args.m, 1.0 / args.m))
    row["proportional"] = var(proportional(mk.parent, idx))
    w, info = race(mk.parent, idx)
    row["race"], row["calib race"] = var(w), info
    row["oracle"] = var(long_only_min_var(Sub))
    row["parent_top_weight"] = float(mk.parent.max())
    row["parent_names_held"] = int((mk.parent > 1e-8 * mk.parent.max()).sum())

    for T in args.Ts:
        X = mk.panel(rng, T)
        _, V, D = factor_correlation(X, args.k)
        w, info = race(mk.parent, idx, V=V, D=D)
        row[f"race+factor T={T}"] = var(w)
        row[f"calib race+factor T={T}"] = info
        row[f"estimate+solve T={T}"] = var(estimate_and_solve(X, idx))
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
    p.add_argument("--rank", type=int, default=5,
                   help="true number of factors in the index market. A rank-1 "
                        "market has only two directions and decides several "
                        "questions by construction; 5 is the honest default.")
    p.add_argument("--target-corr", dest="target_corr", type=float, default=0.27,
                   help="average pairwise correlation the index market is "
                        "solved to. Fixing eps instead gave 0.10 at rank 3, "
                        "which is not an equity market.")
    p.add_argument("--Ts", type=int, nargs="+", default=[20, 40, 100],
                   help="panel lengths for the rules that use data")
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
    for j, g in enumerate(mine):
        td = time.time()
        rows.append(one_draw(a, g))
        print(f"  shard {a.shard}/{a.shards}  draw {g}  "
              f"({j + 1}/{len(mine)})  {time.time() - td:.1f}s", flush=True)

    out = {"config": vars(a), "rows": rows,
           "elapsed_seconds": time.time() - t0,
           "host": platform.node(), "python": platform.python_version(),
           "numpy": np.__version__,
           # which winning actually produced these numbers, not which one pip
           # reports. On the machine this was written, pip showed 1.2.0 from
           # site-packages while the import resolved to a git checkout at
           # 1.5.0, so the installed version is not evidence of anything.
           "winning": getattr(winning, "__version__", "unknown"),
           "winning_path": os.path.dirname(winning.__file__),
           "threads": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
    dest.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {dest}  ({len(rows)} draws, {out['elapsed_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
