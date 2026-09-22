"""Does the trust parameter still behave at five thousand assets?

The companion result says the tilt's dial is a trust parameter: harmful when
the estimated correlation is noise, valuable when it is not, with the
crossover near a third of an observation per asset. Two years of weekly data
on five thousand names is 0.021, fifty times below that crossover, so the
prediction is that phi near zero is correct and that the damage grows
monotonically with phi.

Paired: every phi races the same fixed seed ensemble against the same
benchmark, so what is measured is the dial and not the simulation.
"""
import numpy as np
from allocation.thurstone import ThurstonePortfolio
from huge_universe import market, true_risk, sample, hrp_big

PHIS = (0.0, 0.25, 0.5, 0.75, 1.0)


def run(n=5000, T=104, draws=12, seed=31):
    rng = np.random.default_rng(seed)
    out = {p: [] for p in PHIS}
    base = []
    for i in range(draws):
        B, d = market(rng, n)
        X = sample(rng, B, d, T)
        w_hrp, _ = hrp_big(None, X, 50)
        base.append(true_risk(w_hrp, B, d))
        for p in PHIS:
            w = ThurstonePortfolio(target=w_hrp, factors=3, n_paths=65536,
                                   phi=p).fit(X).weights_
            out[p].append(true_risk(np.asarray(w, float), B, d))
        if (i + 1) % 5 == 0:
            print(f"    draw {i+1}/{draws}", flush=True)
    return {p: np.asarray(v) for p, v in out.items()}, np.asarray(base)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 104
    print(f"n = {n}, T = {T}, T/n = {T/n:.3f}. Tilting HRP.\n")
    out, base = run(n=n, T=T)
    ref = out[0.0]
    print(f"{'phi':>6s}{'variance':>12s}{'ratio to phi=0':>16s}{'better than phi=0':>20s}")
    for p in PHIS:
        x = out[p]
        print(f"{p:6.2f}{np.median(x):12.5f}{np.median(x/ref):16.3f}{np.mean(x < ref):19.0%}")
    print(f"\n  exact HRP {np.median(base):.5f}; the phi=0 race reproduces it to "
          f"{np.median(np.abs(ref - base) / base):.1%}")
    mono = all(np.median(out[PHIS[i]]) <= np.median(out[PHIS[i+1]]) + 1e-12
               for i in range(len(PHIS)-1))
    print(f"  damage increases monotonically in phi: {mono}")
