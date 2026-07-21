"""Generate the embedded data for the high-dimension (n vs T) demo.

A known-truth market: n = 500 names in 10 sectors, one market factor plus one
sector factor each, so the true covariance -- and hence the true minimum-variance
portfolio and its variance -- is known exactly. That denominator is the point of
using synthetic data: every "x true GMV" number on the page is exact.

One simulated history of 2000 days; the estimation window T is the first T rows
(a shrinking window of the same data). Per T the script emits:
  MVOOS, MVLEV : out-of-sample variance ratio and leverage of the sample
                 minimum-variance portfolio  w = S^-1 1 / (1' S^-1 1)
  MVW          : its weights (for the sorted-weight panel)
  LOB, LOS     : rank-12 factor decomposition (loadings, idio std) of the sample
                 correlation, feeding the browser's live low-rank race
plus the true loadings/idio (for exact OOS evaluation of any weight vector in
the browser) and reference variances.

Emits docs/demos/high-dim/data.js. The in-browser race is the forward pass of
transport_weights_lowrank on a fixed common-seed ensemble.
"""
import json, os, numpy as np
from numpy.random import default_rng
from allocation._thurstone.covariance import cov_to_corr, factor_decompose
from allocation._thurstone.transport import transport_weights_lowrank

N, NSEC, K = 500, 10, 12
TS = [2000, 1500, 1200, 1000, 850, 750, 650, 600, 550, 525, 505, 475, 450, 400, 350, 300]

rng = default_rng(0)
bm = rng.uniform(0.15, 0.35, N)                    # market loadings
sec = np.repeat(np.arange(NSEC), N // NSEC)
bs = rng.uniform(0.25, 0.45, N)                    # sector loadings
d = rng.uniform(0.4, 1.0, N)                       # idiosyncratic std
Bt = np.zeros((N, 1 + NSEC)); Bt[:, 0] = bm
for i in range(N):
    Bt[i, 1 + sec[i]] = bs[i]
Sig = Bt @ Bt.T + np.diag(d ** 2)

gmv = np.linalg.solve(Sig, np.ones(N)); gmv /= gmv.sum()
v_gmv = float(gmv @ Sig @ gmv)
eq = np.ones(N) / N
v_eq = float(eq @ Sig @ eq)
print(f"true GMV var {v_gmv:.5f} (leverage {np.abs(gmv).sum():.2f}); equal weight {v_eq/v_gmv:.2f}x")

X = rng.multivariate_normal(np.zeros(N), Sig, max(TS))

MVOOS, MVLEV, MVW, LOB, LOS = [], [], [], [], []
# verification only: the page recomputes race weights live on its own seeds
sf = default_rng(1).standard_normal((1 << 13, K))
si = default_rng(2).standard_normal((1 << 13, N))
for T in TS:
    S = np.cov(X[:T].T)
    w = np.linalg.solve(S, np.ones(N)); w = w / w.sum()
    MVOOS.append(round(float(w @ Sig @ w) / v_gmv, 2))
    MVLEV.append(round(float(np.abs(w).sum()), 2))
    MVW.append([round(float(v), 5) for v in w])
    B, dv = factor_decompose(cov_to_corr(S), K, seed=0)
    LOB.append([[round(float(v), 3) for v in row] for row in B])
    LOS.append([round(float(v), 3) for v in np.sqrt(np.clip(dv, 0.0, None))])
    wr = transport_weights_lowrank(np.zeros(N), B, dv, sf, si)
    print(f"T={T:5d} (n/T={N/T:4.2f}): minvar OOS {MVOOS[-1]:8.2f}x lev {MVLEV[-1]:6.2f} | "
          f"race OOS {float(wr @ Sig @ wr)/v_gmv:.2f}x")

data = {"n": N, "k": K, "Ts": TS,
        "v_gmv": round(v_gmv, 6), "eq_ratio": round(v_eq / v_gmv, 3),
        "Bt": [[round(float(v), 4) for v in row] for row in Bt],
        "dt": [round(float(v), 4) for v in d],
        "MVOOS": MVOOS, "MVLEV": MVLEV, "MVW": MVW, "LOB": LOB, "LOS": LOS}
out = os.path.join(os.path.dirname(__file__), "..", "docs", "demos", "high-dim", "data.js")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    f.write("const HD = " + json.dumps(data, separators=(",", ":")) + ";\n")
print(f"wrote {os.path.relpath(out)}  ({os.path.getsize(out)//1024} KB)")
