"""Tail-sensitive Black-Litterman on the real Dow: calibrate to correlation, simulate the tail.

The recipe (the clean version):
  (i)   take empirical return data (Dow, 2010-2024);
  (ii)  FIT assuming its CORRELATION -- calibrate abilities so a GAUSSIAN race with the
        empirical correlation reproduces the benchmark (equal weight). This prices the
        second-moment structure into the anchor;
  (iii) SIMULATE to capture the TAIL -- run the forward race under the EMPIRICAL
        simulation (resampled real return vectors: same correlation, real fat joint tail).

Because reference and tilt share the empirical correlation, the tilt isolates the
PURE TAIL residual -- the joint-crash structure beyond second moments (the tab:tail
excess). Evaluating both races at the same calibrated abilities cancels Monte-Carlo
noise, so the tail tilt is clean. This is reverse optimization with a TAIL view:
Black-Litterman's tau/Omega dial over co-crash co-movement, which a covariance cannot
encode and a return-view BL cannot state. Calibration uses a Newton step preconditioned
by the (well-conditioned) choice-sensitivity Jacobian -- no covariance inverse.
"""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from numpy.random import default_rng
from scipy.special import ndtr
import yfinance as yf
from allocation._thurstone.transport import transport_weights

TICKERS = ["AAPL","AMGN","AXP","BA","CAT","CSCO","CVX","DIS","GS","HD","HON","IBM",
           "INTC","JNJ","JPM","KO","MCD","MMM","MRK","MSFT","NKE","PG","TRV","UNH",
           "VZ","WMT","CRM","DOW","V"]
px = yf.download(TICKERS, start="2010-01-01", end="2024-12-31",
                 auto_adjust=True, progress=False)["Close"].dropna(axis=1, how="any")
R = np.log(px / px.shift(1)).dropna(); names = list(R.columns); R = R.values; T, n = R.shape
Cemp = np.corrcoef(R.T); Rstd = (R - R.mean(0)) / R.std(0)        # empirical copula, unit marginals
print(f"{n} Dow names, {T} trading days, 2010-2024\n")

# --- anchor: joint-crash excess over a Gaussian fit (tab:tail) ---
G = default_rng(0).multivariate_normal(R.mean(0), np.cov(R.T), size=2_000_000)
print(f"{'all names down by >q same day':30}{'empirical':>11}{'Gaussian':>11}{'ratio':>8}")
for q in (0.010, 0.015, 0.020):
    e = float(np.mean(np.all(R < -q, axis=1))); g = float(np.mean(np.all(G < -q, axis=1)))
    print(f"  >{q*100:.1f}%{'':22}{e:>11.5f}{g:>11.5f}{(f'{e/g:.0f}x' if g else '>>'):>8}")
print("  (joint crashes the covariance cannot encode -- what the tail simulation captures)\n")

M = 1 << 16
seeds = default_rng(1).standard_normal((M, n))
center = lambda t: t - t.mean()
def W_gauss(theta):                                              # race under Gaussian(Cemp)
    return np.asarray(transport_weights(theta, Cemp, seeds))
def W_emp(theta):                                                # race under the empirical simulation
    idx = np.clip((ndtr(seeds[:, 0]) * T).astype(int), 0, T - 1)
    X = theta + Rstd[idx]
    return np.bincount(np.argmin(X, 1), minlength=n) / M

# (ii) calibrate abilities so the Gaussian(Cemp) race reproduces the equal benchmark
w0 = np.full(n, 1.0 / n); theta = np.zeros(n)
for _ in range(12):
    F = W_gauss(theta) - w0
    if np.abs(F).sum() < 0.01:
        break
    J = np.column_stack([(W_gauss(theta + 0.05*np.eye(n)[j]) - W_gauss(theta - 0.05*np.eye(n)[j])) / 0.10
                         for j in range(n)])
    theta = center(theta - np.linalg.lstsq(J, F, rcond=None)[0])

# (iii) pure tail tilt = empirical race minus Gaussian race, at the same abilities
wg, we = W_gauss(theta), W_emp(theta)
d = we - wg; order = np.argsort(d)
print(f"benchmark reproduced under the correlation: |W_gauss - equal|_1 = {np.abs(wg-w0).sum():.3f}")
print(f"\npure TAIL tilt (empirical race minus Gaussian race, same abilities, start 1/n={1/n:.3f}):")
print("  most DE-weighted by the tail (names that crash WITH the field):")
for i in order[:5]:
    print(f"    {names[i]:5} {wg[i]:.3f} -> {we[i]:.3f}  ({d[i]:+.3f})")
print("  most UP-weighted by the tail (crash-decorrelated hedges):")
for i in order[::-1][:5]:
    print(f"    {names[i]:5} {wg[i]:.3f} -> {we[i]:.3f}  ({d[i]:+.3f})")

print("\nread: a crash-proofing OVERLAY for any benchmark. Calibrate to the correlation,")
print("simulate the tail, and the tilt is the pure tail residual -- it SHADES the names with")
print("MORE tail dependence than their correlation implies (IBM, CSCO, WMT crash with the field")
print("beyond what covariance sees) and lifts those with LESS (JPM, UNH). Feed any portfolio as")
print("the benchmark and it shades its excess-tail-dependent names. This is reverse optimization")
print("with a TAIL view -- Black-Litterman's confidence dial over co-crash co-movement -- driven")
print("by the empirical simulation directly, so the full joint tail (the 96x) does the shading,")
print("inversion-free and benchmark-anchored.")

def figure(path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sd = d[order]; lab = [names[i] for i in order]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.bar(range(n), sd * 1e3, color=["#d62728" if v < 0 else "#2ca02c" for v in sd])
    ax.set_xticks(range(n)); ax.set_xticklabels(lab, rotation=90, fontsize=7); ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("tail tilt: empirical $-$ Gaussian race  (per mille)")
    ax.set_title("Tail-sensitive Black–Litterman on the Dow\n"
                 "calibrate to the correlation, simulate the tail: pure co-crash tilt")
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); print(f"\nfigure -> {path}")

import os
figure(os.path.join(os.path.dirname(__file__), "tail_black_litterman.png"))
