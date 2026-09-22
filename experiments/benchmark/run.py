"""Benchmark every allocator in the package on simulated markets.

    python run.py [draws] [market] [reference]

market: gaussian | ar | student-t | regime      reference: any method name
"""
import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import markets
from methods import registry
from protocol import evaluate, summarise

RATIOS = (0.5, 1.0, 2.0, 5.0)

FACTORIES = {
    "gaussian": lambda rng, n: markets.gaussian(rng, n),
    "ar": lambda rng, n: markets.gaussian(rng, n, ar=0.3),
    "student-t": lambda rng, n: markets.student_t(rng, n),
    "regime": lambda rng, n: markets.regime(rng, n),
}

if __name__ == "__main__":
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    which = sys.argv[2] if len(sys.argv) > 2 else "gaussian"
    ref = sys.argv[3] if len(sys.argv) > 3 else "equal weight"
    meth = registry()
    print(f"market: {which}   draws: {draws}   assets: 40   reference: {ref}")
    spread = markets.elliptical_spread(FACTORIES[which](np.random.default_rng(1), 40),
                                       np.random.default_rng(2))
    print(f"elliptical spread of this market: {spread:.3f}  "
          f"({'tail metric adds information' if spread > 1.05 else 'tail metric is a relabelling of variance'})")
    var, es, labels = evaluate(meth, FACTORIES[which], RATIOS, draws,
                               progress=max(draws // 4, 1))
    summarise(var, meth, RATIOS, ref, "true-covariance variance")
    if spread > 1.05:
        summarise(es, meth, RATIOS, ref, "expected shortfall at 5%")
    from collections import Counter
    print("\nmarket families drawn: " + ", ".join(f"{k.split('/')[1]} {v}"
          for k, v in Counter(labels).most_common()))
