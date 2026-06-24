"""Computational cost: redundancy-aware credit via the race vs via Shapley.

All three target a symmetric, efficient, redundancy-aware credit split over n
contributors on the expected-max game v(S)=E[max_{i in S} X_i]:

  exact Shapley   : enumerate 2^n coalitions               -> O(2^n) evaluations
  sampled Shapley : R random permutations, running-max margins -> O(R * M * n)
  race (win-prob) : one argmax sweep gives ALL n shares     -> O(M * n), single pass

The race is not computing the Shapley value (it is the gradient of the same
potential, a cousin -- see shapley_bridge.py); the point is that *if you want a
redundancy-aware attribution*, the race delivers one in a single linear pass where
Shapley is exponential (exact) or needs R>>1 nested passes (sampled).
"""
import time, itertools, math, numpy as np
from numpy.random import default_rng

def make_X(n, M, seed=0):
    rng = default_rng(seed)                                 # 1-factor correlation
    return 0.6 * rng.standard_normal((M, 1)) + rng.standard_normal((M, n))

def race_credit(X):                                          # O(M*n), one pass
    return np.bincount(X.argmax(1), minlength=X.shape[1]) / len(X)

def shapley_exact(X):                                        # O(2^n) coalitions
    n = X.shape[1]
    vmax = lambda S: float(X[:, list(S)].max(1).mean()) if S else 0.0
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]; m = len(others)
        for r in range(m + 1):
            c = math.factorial(r) * math.factorial(m - r) / math.factorial(m + 1)
            for S in itertools.combinations(others, r):
                phi[i] += c * (vmax(S + (i,)) - vmax(S))
    return phi

def shapley_perm(X, R, seed=1):                              # O(R*M*n), running-max margins
    n, M = X.shape[1], len(X); rng = default_rng(seed); phi = np.zeros(n)
    for _ in range(R):
        runmax = np.full(M, -1e18)
        for k in rng.permutation(n):
            xk = X[:, k]
            phi[k] += np.maximum(xk - runmax, 0.0).mean()    # E[(X_k - max_S)^+]
            runmax = np.maximum(runmax, xk)
    return phi / R

def t(fn, *a, reps=3):
    best = float("inf")
    for _ in range(reps):
        s = time.perf_counter(); fn(*a); best = min(best, time.perf_counter() - s)
    return best

# --- A. head-to-head at n=12 (exact feasible): cost + that the race agrees in spirit
print("=== A. n=12 head-to-head (M=20000) ===")
X = make_X(12, 20000)
te = t(lambda: shapley_exact(X), reps=1)
phi = shapley_exact(X); phi /= phi.sum()
tr = t(lambda: race_credit(X))
tp = t(lambda: shapley_perm(X, 200), reps=1); php = shapley_perm(X, 200); php /= php.sum()
print(f"  exact Shapley : {te*1e3:8.1f} ms")
print(f"  sampled (R200): {tp*1e3:8.1f} ms   ||sampled-exact share||_inf = {np.max(np.abs(php-phi)):.4f}")
print(f"  race          : {tr*1e3:8.1f} ms   ({te/tr:.0f}x faster than exact, {tp/tr:.0f}x than sampled)")

# --- B. scaling
print("\n=== B. scaling: wall time vs n ===")
print(f"  {'n':>5} | {'exact 2^n':>12} | {'race O(Mn)':>12}   (M=20000; exact M=1500)")
print("  " + "-" * 50)
for n in [6, 8, 10, 12, 14, 16, 18]:
    Xs = make_X(n, 1500)
    te = t(lambda: shapley_exact(Xs), reps=1) if n <= 18 else None
    Xr = make_X(n, 20000)
    tr = t(lambda: race_credit(Xr))
    es = f"{te*1e3:10.1f} ms" if te is not None else "  (skipped)"
    print(f"  {n:>5} | {es:>12} | {tr*1e3:10.2f} ms")

print("\n  race where exact Shapley is hopeless (M=20000):")
for n in [100, 1000, 5000]:
    Xr = make_X(n, 20000)
    tr = t(lambda: race_credit(Xr))
    print(f"    n={n:>5}: race {tr*1e3:7.2f} ms   (exact Shapley would need 2^{n} coalitions)")

# --- C. headline extrapolation
Xe = make_X(18, 1500); t18 = t(lambda: shapley_exact(Xe), reps=1)
print(f"\n=== C. headline ===")
print(f"  exact Shapley at n=18 took {t18:.2f} s; each +1 in n ~doubles it.")
for n in [30, 40, 60]:
    yrs = t18 * (2 ** (n - 18)) / (3600 * 24 * 365)
    print(f"    extrapolated exact Shapley at n={n}: ~{yrs:.2e} years")
print("  the race returns all n redundancy-aware shares at these n in milliseconds.")

def figure(path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ns = [6, 8, 10, 12, 14, 16, 18]
    ex = [t(lambda: shapley_exact(make_X(n, 1500)), reps=1) for n in ns]
    nr = [6, 8, 10, 12, 14, 16, 18, 32, 64, 128, 256, 512, 1024]
    rc = [t(lambda: race_credit(make_X(n, 20000))) for n in nr]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(ns, ex, "-o", color="#d62728", label="exact Shapley  O(2^n)")
    ax.semilogy(nr, rc, "-o", color="#2ca02c", label="race (win-prob)  O(M·n)")
    ax.set_xlabel("n contributors"); ax.set_ylabel("wall time (s, log)")
    ax.set_title("Redundancy-aware credit: race vs exact Shapley")
    ax.legend(frameon=False); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"\nfigure -> {path}")

import os
figure(os.path.join(os.path.dirname(__file__), "shapley_complexity.png"))
