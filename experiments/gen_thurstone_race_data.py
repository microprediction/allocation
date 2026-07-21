"""Generate the embedded data for the interactive Thurstone race demo.

Emits docs/demos/thurstone-race/data.js with:
  names : a 12-name Dow subset (deliberate sector pairs: tech, banks, pharma, staples)
  C     : empirical daily-return correlation
  vol   : annualized volatilities (for the min-variance comparison covariance)
  theta : abilities calibrated so a Gaussian race under C reproduces equal weight
The browser does the rest live: it builds the clone-augmented correlation, takes a
Cholesky factor, and re-runs the race on a fixed seed ensemble (the common-seed
transport, in JavaScript) each time the clone correlation slider moves.
"""
import json, os, warnings, numpy as np
warnings.filterwarnings("ignore")
from numpy.random import default_rng
import yfinance as yf
from allocation._thurstone.transport import transport_weights

TICKERS = ["AAPL", "MSFT", "CSCO", "JPM", "GS", "CVX", "JNJ", "MRK",
           "KO", "PG", "CAT", "WMT"]
px = yf.download(TICKERS, start="2010-01-01", end="2024-12-31",
                 auto_adjust=True, progress=False)["Close"].dropna(axis=1, how="any")
px = px[TICKERS]  # keep the deliberate ordering (sector pairs adjacent)
R = np.log(px / px.shift(1)).dropna()
names = list(R.columns)
R = R.values
n = len(names)
C = np.corrcoef(R.T)
vol = R.std(0) * np.sqrt(252)

# calibrate abilities so the Gaussian race under C is equal-weight (Newton on the
# forward race map, same seeds throughout)
M = 1 << 16
seeds = default_rng(1).standard_normal((M, n))
center = lambda t: t - t.mean()
W = lambda theta: np.asarray(transport_weights(theta, C, seeds))

w0 = np.full(n, 1.0 / n)
theta = np.zeros(n)
for _ in range(12):
    F = W(theta) - w0
    if np.abs(F).sum() < 0.005:
        break
    J = np.column_stack([(W(theta + 0.05 * np.eye(n)[j]) - W(theta - 0.05 * np.eye(n)[j])) / 0.10
                         for j in range(n)])
    theta = center(theta - np.linalg.lstsq(J, F, rcond=None)[0])
print(f"calibrated: |W - equal|_1 = {np.abs(W(theta) - w0).sum():.4f}")

data = {"names": names, "n": n,
        "theta": [round(float(x), 4) for x in theta],
        "vol": [round(float(v), 4) for v in vol],
        "C": [[round(float(v), 4) for v in row] for row in C]}
out = os.path.join(os.path.dirname(__file__), "..", "docs", "demos", "thurstone-race", "data.js")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    f.write("const TR = " + json.dumps(data, separators=(",", ":")) + ";\n")
print(f"wrote {os.path.relpath(out)}  ({os.path.getsize(out)} bytes)")
