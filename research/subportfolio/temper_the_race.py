"""The race selects names well and stops concentrating too early. Fix the second.

The concentration-matched control separated two things the earlier comparison
had merged. At the race's own breadth, race+factor beats the parent's ordering
on every draw by 2.2 percent, so the correlation it uses is real information.
But the race holds 34 effective names where the sub-universe optimum holds 92,
and equal weight at 200 beats both. The selection is good and the breadth is
wrong, and the breadth error is the larger one.

So temper the race's OWN weights rather than the parent's, and sweep the
target breadth. If selection and breadth are separable, race+factor tempered
toward the optimum's breadth should beat equal weight, which has the breadth
roughly right and no selection at all.

The breadth target is expressed as a fraction of the sub-universe, because an
index holder knows m and does not know the oracle.
"""
import sys
import numpy as np

from markets import IndexMarket
from rules import long_only_min_var, factor_correlation, proportional, race
from tempering_control import temper, eff


def run(n=2000, m=200, draws=10, seed=31, T=104, k=2,
        fracs=(0.17, 0.30, 0.45, 0.60, 0.80, 1.00)):
    rng = np.random.default_rng(seed)
    out = {f"race+factor @ {f:.0%}": [] for f in fracs}
    out.update({"proportional": [], "race+factor": [], "equal weight": [],
                "oracle": [], "oracle breadth": []})
    for _ in range(draws):
        mk = IndexMarket(rng, n, rank=5, target_corr=0.27)
        idx = np.sort(rng.choice(n, m, replace=False))
        Sub = mk.block(idx)
        v = lambda w: float(np.asarray(w, float) @ Sub @ np.asarray(w, float))
        X = mk.panel(rng, T)
        _, V, D = factor_correlation(X, k)
        pw = proportional(mk.parent, idx)
        rf = race(mk.parent, idx, V=V, D=D)[0]
        orc = long_only_min_var(Sub)
        out["proportional"].append(v(pw))
        out["race+factor"].append(v(rf))
        out["equal weight"].append(v(np.full(m, 1.0 / m)))
        out["oracle"].append(v(orc))
        out["oracle breadth"].append(eff(orc) / m)
        for f in fracs:
            out[f"race+factor @ {f:.0%}"].append(v(temper(rf, f * m)))
    return out, fracs


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    out, fracs = run(n=n, m=m)
    base = np.array(out["proportional"])
    ew = np.array(out["equal weight"])
    nd = len(base)
    print(f"\nparent {n}, sub-universe {m}, {nd} draws. The oracle's own breadth "
          f"is {np.median(out['oracle breadth']):.0%} of the sub-universe.\n")
    print(f"{'rule':24s}{'vs proportional':>17}{'vs equal weight':>17}"
          f"{'beats ew':>10}")
    for key in (["proportional", "race+factor"]
                + [f"race+factor @ {f:.0%}" for f in fracs]
                + ["equal weight", "oracle"]):
        x = np.array(out[key])
        print(f"{key:24s}{np.median(x / base):17.3f}{np.median(x / ew):17.3f}"
              f"{np.mean(x < ew):10.0%}")
