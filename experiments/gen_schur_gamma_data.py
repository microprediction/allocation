"""Generate the embedded data for the interactive Schur gamma demo.

Emits docs/demos/schur-gamma/data.js with:
  names     : 28 Dow tickers (2010-2024 daily returns)
  corr      : empirical correlation (for the heatmap)
  orderF    : Fiedler seriation order (allocation._schur.seriation.seriate)
  orderD    : agglomerative dendrogram leaf order (classic HRP ordering)
  gammas, Wg, effg, varr :
              Schur weights on a gamma grid under the Fiedler order
              (compute_monotonic_weights), the effective gamma actually used,
              and portfolio variance relative to unconstrained minimum variance
  years, cumF, cumD :
              cumulative L1 turnover of a weekly-rebalanced gamma=0.5 portfolio
              under (F) warm-started Fiedler seriation and (D) a dendrogram
              order recomputed each week, same EWMA covariance for both
"""
import json, os, warnings, numpy as np
warnings.filterwarnings("ignore")
import yfinance as yf
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from allocation._schur.seriation import seriate
from allocation._schur.coupling import compute_monotonic_weights
from allocation._thurstone.covariance import cov_to_corr

TICKERS = ["AAPL","AMGN","AXP","BA","CAT","CSCO","CVX","DIS","GS","HD","HON","IBM",
           "INTC","JNJ","JPM","KO","MCD","MMM","MRK","MSFT","NKE","PG","TRV","UNH",
           "VZ","WMT","CRM","DOW","V"]
px = yf.download(TICKERS, start="2010-01-01", end="2024-12-31",
                 auto_adjust=True, progress=False)["Close"].dropna(axis=1, how="any")
R = np.log(px / px.shift(1)).dropna()
names = list(R.columns); dates = R.index
Rv = R.values; T, n = Rv.shape
cov = np.cov(Rv.T) * 252
corr = np.corrcoef(Rv.T)

def dendro_order(c):
    d = np.sqrt(np.clip(0.5 * (1.0 - c), 0.0, 1.0))
    np.fill_diagonal(d, 0.0)
    return leaves_list(linkage(squareform(d, checks=False), method="average"))

orderF, _ = seriate(cov)
orderD = dendro_order(corr)

# ---- weights over a gamma grid (Fiedler order, monotonic sweep) -------------
gmv = np.linalg.solve(cov, np.ones(n)); gmv /= gmv.sum()
var_gmv = float(gmv @ cov @ gmv)
gammas = np.round(np.linspace(0.0, 1.0, 51), 2)
Wg, effg, varr = [], [], []
for g in gammas:
    w, eg = compute_monotonic_weights(orderF, cov, float(g))
    Wg.append([round(float(x), 4) for x in w])
    effg.append(round(float(eg), 3))
    varr.append(round(float(w @ cov @ w) / var_gmv, 3))
print(f"gamma sweep: var ratio {varr[0]:.3f} (HRP) -> {varr[-1]:.3f} (gamma=1); "
      f"effective gamma at 1.0: {effg[-1]:.3f}")

# ---- weekly turnover: warm-started Fiedler vs recomputed dendrogram ---------
HL = 60.0
lam = 0.5 ** (1.0 / HL)
Sig = np.cov(Rv[:252].T)          # burn-in year initializes the EWMA
mu = Rv[:252].mean(0)
GAMMA = 0.5
FREQ = 5
prevF = prevD = None; vF = None
years, tF, tD = [], [], []
for t in range(252, T):
    x = Rv[t]
    mu = lam * mu + (1 - lam) * x
    d = x - mu
    Sig = lam * Sig + (1 - lam) * np.outer(d, d)
    if (t - 252) % FREQ:
        continue
    oF, vF = seriate(Sig, previous=vF)
    wF, _ = compute_monotonic_weights(oF, Sig, GAMMA)
    oD = dendro_order(cov_to_corr(Sig))
    wD, _ = compute_monotonic_weights(oD, Sig, GAMMA)
    if prevF is not None:
        years.append(round(dates[t].year + (dates[t].dayofyear - 1) / 365.25, 3))
        tF.append(float(np.abs(wF - prevF).sum()))
        tD.append(float(np.abs(wD - prevD).sum()))
    prevF, prevD = wF, wD

cumF = np.cumsum(tF); cumD = np.cumsum(tD)
print(f"turnover over {len(tF)} rebalances: Fiedler total {cumF[-1]:.2f}, "
      f"dendrogram total {cumD[-1]:.2f} (x{cumD[-1]/cumF[-1]:.1f})")

data = {"names": names, "n": n,
        "corr": [[round(float(v), 3) for v in row] for row in corr],
        "orderF": [int(i) for i in orderF],
        "orderD": [int(i) for i in orderD],
        "gammas": [float(g) for g in gammas],
        "Wg": Wg, "effg": effg, "varr": varr,
        "years": years,
        "cumF": [round(float(v), 3) for v in cumF],
        "cumD": [round(float(v), 3) for v in cumD]}
out = os.path.join(os.path.dirname(__file__), "..", "docs", "demos", "schur-gamma", "data.js")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    f.write("const SG = " + json.dumps(data, separators=(",", ":")) + ";\n")
print(f"wrote {os.path.relpath(out)}  ({os.path.getsize(out)//1024} KB)")
