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
from rules import (long_only_min_var, long_only_max_sharpe, long_only_min_cvar,
                   expected_shortfall, factor_correlation, factor_covariance,
                   proportional, race, flattened, black_litterman,
                   race_regime, estimate_and_solve)

PREMISE_TOL = 1e-9      # the premise holds by construction; this catches bugs


def sub_universe(mk, rng, m, how):
    """The survivors. `random` is m names drawn uniformly. `sector` takes whole
    sectors, in a random order, until at least m names are in hand, then keeps
    the first m: a sector sub-index. Random sub-universes lose little
    diversification and have little at stake; sector ones are what index
    providers actually build."""
    if how == "random":
        return np.sort(rng.choice(mk.n, m, replace=False))
    take = []
    for s in rng.permutation(mk.sectors):
        take.extend(np.flatnonzero(mk.sector == s).tolist())
        if len(take) >= m:
            break
    return np.sort(np.array(take[:m]))


def one_draw(args, g):
    """Run global draw index g. Returns a dict of method -> variance."""
    rng = np.random.default_rng([args.seed, g])
    Market = MARKETS[args.scale]
    kw = {"law": args.law} if args.scale == "index" else {}
    mk = Market(rng, args.n, solver=long_only_min_var, **kw)

    resid = mk.premise_residual()
    if resid > PREMISE_TOL:
        return {"draw": g, "skipped": "premise", "premise_residual": resid}

    idx = sub_universe(mk, rng, args.m, args.subset)
    Sub = mk.block(idx)
    m_S = mk.m[idx]
    ev = np.linalg.eigvalsh(Sub)

    # Two scores per rule. Sharpe is exact from (Sigma, m). Expected shortfall
    # is under the true LAW, from a scenario sample of the sub-universe, which
    # is where a regime the covariance cannot carry shows up.
    scen = mk.panel(np.random.default_rng([args.seed, g, 7]), args.scenarios, idx)
    row_es = {}

    def score(w, name):
        w = np.asarray(w, float)
        es = expected_shortfall(scen, w)
        row_es["es95 " + name] = es
        row_es["starr " + name] = float(w @ m_S) / es      # return per unit of shortfall
        return float(w @ m_S) / float(np.sqrt(w @ Sub @ w))

    row = {"draw": g, "family": mk.family, "objective": "sharpe",
           "premise_residual": resid,
           "min_eig": float(ev.min()), "parent_effective_n":
           float(1.0 / np.sum(mk.parent ** 2))}
    row["equal weight"] = score(np.full(args.m, 1.0 / args.m), "equal weight")
    row["proportional"] = score(proportional(mk.parent, idx), "proportional")
    row["flattened"] = score(flattened(mk.parent, idx), "flattened")
    row["race"] = score(race(mk.parent, idx), "race")
    row["oracle"] = score(long_only_max_sharpe(Sub, m_S), "oracle")
    row["tail oracle"] = score(long_only_min_cvar(scen, m_S), "tail oracle")

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
            row[f"race+factor T={T}"] = score(race(mk.parent, idx, V=V, D=D), f"race+factor T={T}")
            if getattr(mk, "pc", 0.0) > 0:
                name = f"race+regime T={T}"
                try:
                    row[name] = score(race_regime(mk.parent, idx, V, D, mk.regime()), name)
                except ValueError as err:       # recorded, not hidden
                    row[name] = row_es["es95 " + name] = row_es["starr " + name] = float("nan")
                    row.setdefault("failed", []).append(f"{name}: {err}")
        if T in set(est_Ts):
            row[f"estimate+solve T={T}"] = score(estimate_and_solve(X, idx), f"estimate+solve T={T}")
            sd, Vc, Dc = factor_covariance(X, args.k)
            row[f"black-litterman T={T}"] = score(
                black_litterman(mk.parent, idx, sd, Vc, Dc), f"black-litterman T={T}")
    row.update(row_es)
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
    p.add_argument("--subset", choices=["random", "sector"], default="random",
                   help="how the sub-universe is chosen")
    p.add_argument("--law", choices=["gaussian", "regime"], default="gaussian",
                   help="the true law: gaussian, or gaussian plus crash and cascade days")
    p.add_argument("--scenarios", type=int, default=20000,
                   help="true-law scenarios for scoring expected shortfall; the "
                        "tail oracle is fitted on the same sample, so it is an "
                        "in-sample bound by construction")
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
    tag = a.tag or f"{a.scale}-{a.subset}-{a.law}-n{a.n}-m{a.m}-k{a.k}-s{a.seed}"
    mine = [g for g in range(a.draws) if g % a.shards == a.shard]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    dest = Path(a.out) / f"{tag}-shard{a.shard}of{a.shards}.json"

    rows, t0 = [], time.time()
    # Resume. A shard file that already exists is a partial run of this same
    # shard, so its draws are kept and skipped, not recomputed and not
    # overwritten. Restarting with the same shard count is then free.
    if dest.exists():
        try:
            rows = [r for r in json.loads(dest.read_text())["rows"]
                    if r["draw"] in set(mine)]
        except (ValueError, KeyError):
            rows = []
        done = {r["draw"] for r in rows}
        mine = [g for g in mine if g not in done]
        if done:
            print(f"  shard {a.shard}/{a.shards}: resuming, {len(done)} draws "
                  f"already on disk, {len(mine)} to go", flush=True)
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
