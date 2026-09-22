"""Does tilting HRP help at five thousand assets and two years of weekly data?

The race is evaluated by quadrature, not simulated, so phi = 0 returns the
benchmark to machine precision and every number below is the dial rather than
sampling noise. The blend stays inside the factor family exactly, so there is
no approximation beyond the factor count.

T/n = 0.021 here, fifty times below the crossover the companion result puts
near a third of an observation per asset, so the prediction is that phi near
zero wins and that the damage grows with phi.
"""
import sys, time
import numpy as np
import winning
from huge_universe import market, sample, true_risk, hrp_big

PHIS = (0.0, 0.1, 0.25, 0.5, 1.0)


def factor_form(X, k=3):
    """A k-factor correlation from the sample, via the T x T gram so the
    n x n matrix is never formed."""
    Xc = X - X.mean(0)
    sd = Xc.std(0, ddof=1); sd[sd == 0] = 1.0
    Z = Xc / sd
    G = (Z @ Z.T) / (len(Z) - 1)
    ev, U = np.linalg.eigh(G)
    idx = np.argsort(ev)[::-1][:k]
    lam = np.maximum(ev[idx], 1e-12)
    V = (Z.T @ U[:, idx]) / np.sqrt(lam) / np.sqrt(len(Z) - 1) * np.sqrt(lam)
    D = np.clip(1.0 - (V ** 2).sum(1), 1e-6, None)
    s = np.sqrt((V ** 2).sum(1) + D)
    return V / s[:, None], D / s ** 2


def race(ability, V=None, D=None):
    p = winning.race_probabilities(ability) if V is None else \
        winning.race_probabilities(ability, V=V, D=D)
    p = np.asarray(p[0] if isinstance(p, tuple) else p, float)
    return p / p.sum()


def run(n=5000, T=104, draws=10, seed=17):
    rng = np.random.default_rng(seed)
    out = {p: [] for p in PHIS}
    base, times = [], {p: [] for p in PHIS}
    for i in range(draws):
        B, d = market(rng, n)
        X = sample(rng, B, d, T)
        w_hrp, _ = hrp_big(None, X, 50)
        base.append(true_risk(w_hrp, B, d))
        a = np.asarray(winning.calibrate_abilities(np.maximum(w_hrp, 1e-12)), float)
        V, D = factor_form(X, 3)
        for phi in PHIS:
            t = time.time()
            w = race(a) if phi == 0.0 else race(a, np.sqrt(phi) * V, (1 - phi) + phi * D)
            times[phi].append(time.time() - t)
            out[phi].append(true_risk(w, B, d))
        print(f"    draw {i+1}/{draws}", flush=True)
    return {p: np.asarray(v) for p, v in out.items()}, np.asarray(base), times


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 104
    draws = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    print(f"n = {n}, T = {T}, T/n = {T/n:.3f}. Tilting HRP, exact race.\n")
    out, base, times = run(n, T, draws)
    print(f"\n{'phi':>6s}{'variance':>12s}{'ratio to HRP':>15s}{'beats HRP':>12s}{'seconds':>10s}")
    for p in PHIS:
        x = out[p]
        print(f"{p:6.2f}{np.median(x):12.5f}{np.median(x / base):15.3f}"
              f"{np.mean(x < base):11.0%}{np.median(times[p]):10.1f}")
    print(f"\n  exact HRP {np.median(base):.5f}")
    print(f"  phi=0 reproduces it to {np.median(np.abs(out[0.0] - base) / base):.2e}")
