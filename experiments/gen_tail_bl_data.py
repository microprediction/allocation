"""Generate the embedded data for the interactive tail-sensitive Black-Litterman demo.

Emits docs/demos/tail-black-litterman/data.js with:
  names : the Dow tickers
  theta : abilities calibrated so a Gaussian race with the empirical correlation
          reproduces the equal-weight benchmark (the in-browser race only does the
          cheap forward pass; this calibration is precomputed here)
  L     : Cholesky factor of the empirical correlation (for the Gaussian draws)
  R     : standardized empirical return vectors (the tail simulation to resample)
The browser races theta against a phi-mixture of Gaussian(L) noise and resampled R
rows, argmin-wins, and reads off the win-frequency weights.
"""
import json, os, warnings, numpy as np
warnings.filterwarnings("ignore")
from numpy.random import default_rng
import yfinance as yf
from allocation._thurstone.transport import transport_weights

TICKERS = ["AAPL","AMGN","AXP","BA","CAT","CSCO","CVX","DIS","GS","HD","HON","IBM",
           "INTC","JNJ","JPM","KO","MCD","MMM","MRK","MSFT","NKE","PG","TRV","UNH",
           "VZ","WMT","CRM","DOW","V"]
px = yf.download(TICKERS, start="2010-01-01", end="2024-12-31",
                 auto_adjust=True, progress=False)["Close"].dropna(axis=1, how="any")
R = np.log(px / px.shift(1)).dropna(); names = list(R.columns); R = R.values; T, n = R.shape
Cemp = np.corrcoef(R.T); L = np.linalg.cholesky(Cemp)
Rstd = (R - R.mean(0)) / R.std(0)

M = 1 << 16
seeds = default_rng(1).standard_normal((M, n))
center = lambda t: t - t.mean()
def W_gauss(theta): return np.asarray(transport_weights(theta, Cemp, seeds))

w0 = np.full(n, 1.0 / n); theta = np.zeros(n)
for _ in range(12):
    F = W_gauss(theta) - w0
    if np.abs(F).sum() < 0.01: break
    J = np.column_stack([(W_gauss(theta + 0.05*np.eye(n)[j]) - W_gauss(theta - 0.05*np.eye(n)[j])) / 0.10
                         for j in range(n)])
    theta = center(theta - np.linalg.lstsq(J, F, rcond=None)[0])
print(f"calibrated: |W_gauss - equal|_1 = {np.abs(W_gauss(theta)-w0).sum():.3f}")

data = {"names": names, "n": n, "T": T,
        "theta": [round(float(x), 4) for x in theta],
        "L": [[round(float(v), 4) for v in row] for row in L],
        "R": [[round(float(v), 2) for v in row] for row in Rstd]}
out = os.path.join(os.path.dirname(__file__), "..", "docs", "demos", "tail-black-litterman", "data.js")
with open(out, "w") as f:
    f.write("const TBL = " + json.dumps(data, separators=(",", ":")) + ";\n")
print(f"wrote {os.path.relpath(out)}  ({os.path.getsize(out)//1024} KB)")
