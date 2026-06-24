"""Race as redundancy-aware variable selection -- vs lasso & stability selection.

The credit/attribution thread suggests the calibrated race as a feature-importance
/ screening rule. The competitor for *stable* selection under collinearity is
stability selection (Meinshausen-Buhlmann: lasso over many subsamples, rank by
selection frequency). Question: under near-duplicate features, does the race give a
stable, redundancy-aware ranking -- copies SHARE credit and don't thrash across
data resamples -- more cheaply?

Setup: one signal direction present as K near-duplicate columns + many noise
columns. A good selector credits the signal cluster, rejects noise, and is stable
across resamples. We compare, over many data seeds:
  lasso        : |LassoCV coef|, normalized            (picks one copy, flips)
  stability    : lasso selection frequency over B subsamples, normalized
  race (calib) : abilities = marginal importance, tilt by feature correlation
We report cluster total, noise total, copy evenness, ACROSS-SEED volatility of the
per-copy credit (the stability metric), and time.
"""
import time, warnings, numpy as np
warnings.filterwarnings("ignore")
from numpy.random import default_rng
from sklearn.linear_model import LassoCV, Lasso
from allocation._thurstone.calibrate import calibrate_diagonal, base_density
from allocation._thurstone.transport import transport_weights

N, K, n_noise, B = 300, 4, 16, 80
SEEDS = 20
nf = (K + 1) + n_noise
sig_idx = list(range(K + 1)); noise_idx = list(range(K + 1, nf))
norm = lambda v: (np.abs(v) / max(np.abs(v).sum(), 1e-12))

def make(seed):
    g = default_rng(seed)
    x = g.standard_normal(N)
    sig = x[:, None] + 0.12 * g.standard_normal((N, K + 1))   # K+1 near-duplicates
    X = np.hstack([sig, g.standard_normal((N, n_noise))])
    y = 3.0 * x + 0.5 * g.standard_normal(N)
    return X, y, g

def lasso_imp(X, y):
    return norm(LassoCV(cv=4, n_jobs=-1).fit(X, y).coef_)

def stability_imp(X, y, g):
    a = LassoCV(cv=4).fit(X, y).alpha_
    freq = np.zeros(nf)
    for _ in range(B):
        idx = g.choice(N, N // 2, replace=False)
        freq += (np.abs(Lasso(alpha=a).fit(X[idx], y[idx]).coef_) > 1e-8)
    return norm(freq / B)

def race_imp(X, y):
    uni = np.array([np.corrcoef(X[:, i], y)[0, 1] ** 2 for i in range(nf)])
    theta = calibrate_diagonal(uni / uni.sum(), base=base_density())
    seeds = default_rng(7).standard_normal((1 << 13, nf))
    return np.asarray(transport_weights(theta, np.corrcoef(X.T), seeds))

methods = {"lasso": None, "stability": None, "race (calib)": None}
acc = {m: [] for m in methods}; tm = {m: 0.0 for m in methods}
for s in range(SEEDS):
    X, y, g = make(s)
    for m, fn in [("lasso", lambda: lasso_imp(X, y)),
                  ("stability", lambda: stability_imp(X, y, g)),
                  ("race (calib)", lambda: race_imp(X, y))]:
        t = time.perf_counter(); v = fn(); tm[m] += time.perf_counter() - t
        acc[m].append(v)

print(f"one signal as {K+1} near-duplicate columns + {n_noise} noise; {SEEDS} data seeds\n")
hdr = f"{'method':14}{'signal':>9}{'noise':>8}{'copy spread':>13}{'copy volatility':>17}{'time/seed':>11}"
print(hdr); print("-" * len(hdr))
for m in methods:
    A = np.array(acc[m])                                  # (SEEDS, nf)
    cluster = A[:, sig_idx].sum(1).mean()
    noise = A[:, noise_idx].sum(1).mean()
    spread = (A[:, sig_idx].max(1) / np.clip(A[:, sig_idx].min(1), 1e-9, None)).mean()
    volatility = A[:, sig_idx].std(0).mean()              # across-seed sd of per-copy credit
    print(f"{m:14}{cluster:>9.3f}{noise:>8.3f}{spread:>13.1f}{volatility:>17.4f}{tm[m]/SEEDS*1e3:>9.0f}ms")
print("\nlower copy spread = more even across duplicates; lower copy volatility = more stable")
print("across data resamples. The race aims for even + stable + cheap; lasso picks one copy")
print("(high spread, high volatility); stability selection is steadier but costs B lasso fits.")
