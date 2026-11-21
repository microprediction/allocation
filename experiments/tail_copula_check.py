"""Table 1 companion (paper: Thurstone portfolios, Section 2.2). Joint all-down / all-up frequencies: empirical vs (a) Gaussian fit to mean+cov,
(b) Gaussian copula with the empirical marginals (normal-scores correlation)."""
import numpy as np, pandas as pd, json
from scipy.stats import norm, rankdata
import sys
if len(sys.argv) > 1 and sys.argv[1].endswith(".pkl"):
    px = pd.read_pickle(sys.argv[1])
else:
    import yfinance as yf
    from dow_study import TICKERS
    px = yf.download(TICKERS, start="2010-01-01", end="2024-12-31", auto_adjust=True,
                     progress=False)["Close"].dropna(axis=1, how="any")
R = np.log(px / px.shift(1)).dropna().values; T, n = R.shape
U = (np.apply_along_axis(rankdata, 0, R) - 0.5) / T
Cns = np.corrcoef(norm.ppf(U).T)
L = np.linalg.cholesky(Cns); Lg = np.linalg.cholesky(np.cov(R.T)); mu = R.mean(0)
rng = np.random.default_rng(0); N, CH = 100_000_000, 2_000_000
qs = (0.010, 0.015, 0.020, 0.030)
lo = {q: norm.ppf(np.clip(np.mean(R < -q, 0), 1e-12, 1)) for q in qs}
hi = {q: norm.ppf(np.clip(1 - np.mean(R > q, 0), 0, 1 - 1e-12)) for q in qs}
cnt = {k: {q: 0 for q in qs} for k in ("cop_down", "cop_up", "gau_down", "gau_up")}
for _ in range(N // CH):
    Z = rng.standard_normal((CH, n)); X = Z @ L.T; G = mu + Z @ Lg.T
    for q in qs:
        cnt["cop_down"][q] += int(np.all(X < lo[q], 1).sum()); cnt["cop_up"][q] += int(np.all(X > hi[q], 1).sum())
        cnt["gau_down"][q] += int(np.all(G < -q, 1).sum()); cnt["gau_up"][q] += int(np.all(G > q, 1).sum())
out = {}
for q in qs:
    out[q] = dict(emp_down=float(np.mean(np.all(R < -q, 1))), emp_up=float(np.mean(np.all(R > q, 1))),
                  **{k: cnt[k][q] / N for k in cnt})
    print(q, {k: f"{v:.3g}" for k, v in out[q].items()}, flush=True)
json.dump({str(k): v for k, v in out.items()},
          open(__import__("os").path.join(__import__("os").path.dirname(__file__), "results", "tail_copula_check.json"), "w"), indent=1)
