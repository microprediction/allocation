"""Bowling as a generative model for assets: continuous returns + contagious default.

The idea (Cotton): read a bowling-alley rigid-body simulation as a market. A pin is a
firm; its position diffuses under collisions (continuous stock movement); the knock-down
threshold is a DEFAULT BARRIER; and the chain reaction is DEFAULT CONTAGION -- one name
falling topples its neighbours. So a single mechanism yields, per scenario:
  - a CONTINUOUS return  (how far the pin was nudged), and
  - a DEFAULT event      (pin displaced past the barrier),
with defaults that CLUSTER through the cascade far beyond what a Gaussian copula with the
same correlations predicts -- the canonical credit tail risk.

Two generalizations folded in:
  (1) continuous + default: every name has a continuous mark-to-market move; crossing the
      barrier is an absorbing jump-to-default. Structural (first-passage) flavour, but the
      contagion is mechanical, not a parameter.
  (2) the setup need not be the bowling triangle: we RANDOMIZE the pin layout, so each
      world is a different market with its own dependence/contagion geometry.

Physics ported from humpday's bowling demo (impulse-based ball-pin and pin-pin collisions).
"""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from numpy.random import default_rng
from scipy.stats import norm

W = 800.0
MB, MP, E_BP, E_PP, PF, BFX, BFY = 10.0, 1.0, 0.5, 0.5, 0.96, 0.991, 0.991
BR, PR, BARRIER = 16.0, 9.0, 55.0          # ball radius, pin radius, default barrier (px displaced)

def random_setup(rng, n=30, k=3):
    """A random market: n firms in k tight clusters with gaps between them. A throw that
    misses leaves everyone standing; a throw INTO a cluster cascades the whole cluster at
    once -- bimodal, all-or-nothing co-default a Gaussian copula cannot represent. (Not the
    traditional triangle: randomizing the setup gives a different market each time.)"""
    cx, cy = rng.uniform(300, 500, k), rng.uniform(200, 330, k)
    per = [n // k + (1 if i < n % k else 0) for i in range(k)]
    X0 = np.concatenate([cx[i] + rng.normal(0, 5, per[i]) for i in range(k)])
    Y0 = np.concatenate([cy[i] + rng.normal(0, 5, per[i]) for i in range(k)])
    return X0, Y0

def decode(u):                              # wide aim so most throws graze, cascades are rare
    return 13 + 10*u[0], -16 + 32*u[1], -3 + 6*u[2], 250 + 300*u[3]

def throw(X0, Y0, u):
    """One scenario: simulate a throw on this setup, return each pin's displacement."""
    n = len(X0)
    speed, angleDeg, spin, releaseX = decode(u); ang = np.radians(angleDeg)
    x, y = X0.copy(), Y0.copy(); vx, vy = np.zeros(n), np.zeros(n)
    bx, by = float(releaseX), 470.0
    bvx, bvy, bsp = speed*np.sin(ang), -speed*np.cos(ang), spin
    for _ in range(260):
        bV2 = bvx*bvx + bvy*bvy; moving = (vx*vx + vy*vy) > 0.04
        if bV2 < 0.04 and not moving.any(): break
        if by < 20 and bV2 < 4 and not moving.any(): break
        bx += bvx; by += bvy; bvx += bsp*0.012; bsp *= 0.985; bvx *= BFX; bvy *= BFY
        if bx < 30: bx, bvx = 30, bvx*-0.55
        if bx > W-30: bx, bvx = W-30, bvx*-0.55
        m = (vx != 0) | (vy != 0)
        x[m] += vx[m]; y[m] += vy[m]; vx[m] *= PF; vy[m] *= PF
        lo, hi = x < 30, x > W-30
        x[lo], vx[lo] = 30, vx[lo]*-0.4; x[hi], vx[hi] = W-30, vx[hi]*-0.4
        for i in range(n):                  # ball-pin
            dx, dy = x[i]-bx, y[i]-by; d2 = dx*dx + dy*dy; md = BR+PR
            if d2 >= md*md or d2 < 1e-6: continue
            d = np.sqrt(d2); nx, ny = dx/d, dy/d
            vrel = (vx[i]-bvx)*nx + (vy[i]-bvy)*ny
            if vrel > 0: continue
            j = -(1+E_BP)*vrel / (1/MB + 1/MP); jx, jy = j*nx, j*ny
            bvx -= jx/MB; bvy -= jy/MB; vx[i] += jx/MP; vy[i] += jy/MP
            ov = md - d + 0.5
            bx -= nx*ov*(MP/(MB+MP)); by -= ny*ov*(MP/(MB+MP))
            x[i] += nx*ov*(MB/(MB+MP)); y[i] += ny*ov*(MB/(MB+MP))
        for i in range(n):                  # pin-pin (the contagion)
            aMov = vx[i]*vx[i] + vy[i]*vy[i] > 1e-4
            for k in range(i+1, n):
                if not aMov and not (vx[k]*vx[k] + vy[k]*vy[k] > 1e-4): continue
                dx, dy = x[k]-x[i], y[k]-y[i]; d2 = dx*dx + dy*dy; md = 2*PR
                if d2 >= md*md or d2 < 1e-6: continue
                d = np.sqrt(d2); nx, ny = dx/d, dy/d
                vrel = (vx[k]-vx[i])*nx + (vy[k]-vy[i])*ny
                if vrel > 0: continue
                j = -(1+E_PP)*vrel / (2/MP)
                vx[i] -= j*nx/MP; vy[i] -= j*ny/MP; vx[k] += j*nx/MP; vy[k] += j*ny/MP
                ov = md - d + 0.5
                x[i] -= nx*ov*0.5; y[i] -= ny*ov*0.5; x[k] += nx*ov*0.5; y[k] += ny*ov*0.5
    return np.hypot(x - X0, y - Y0)

# --- one random market: a panel of continuous returns + default events ---
n, T = 30, 2500
rng = default_rng(7)
X0, Y0 = random_setup(rng, n)
disp = np.array([throw(X0, Y0, rng.random(4)) for _ in range(T)])
default = disp > BARRIER                                   # jump-to-default events
R = -disp + 0.05*rng.standard_normal((T, n))               # continuous mark-to-market loss
Rstd = (R - R.mean(0)) / R.std(0)
pmarg = default.mean(0)                                    # per-name marginal default rate
print(f"random bowling market: {n} firms, {T} scenarios")
print(f"marginal default rate: min {pmarg.min():.1%}, mean {pmarg.mean():.1%}, max {pmarg.max():.1%}")
print(f"mean defaults / scenario: {default.sum(1).mean():.1f}\n")

# --- contagion: simultaneous defaults vs a Gaussian copula with the SAME marginals + corr ---
Ccont = np.corrcoef(Rstd.T)
Z = default_rng(11).multivariate_normal(np.zeros(n), Ccont, size=400_000)
thr = norm.ppf(pmarg)                                      # latent default thresholds
gdef = Z < thr
nb, ng = default.sum(1), gdef.sum(1)
print(f"  P(0 defaults)   bowling {np.mean(nb==0):.3f}   Gauss copula {np.mean(ng==0):.3f}  (more 'all quiet')")
print(f"{'k+ firms default same scenario':32}{'bowling':>10}{'Gauss copula':>14}{'ratio':>8}")
for k in (5, 10, 15, 20):
    e = float(np.mean(nb >= k)); g = float(np.mean(ng >= k))
    print(f"  >= {k:2d}{'':24}{e:>10.4f}{g:>14.4f}{(f'{e/g:.1f}x' if g else '>>'):>8}")
print("  bowling is BIMODAL: mostly all-quiet, punctuated by whole-cluster wipeouts -- so its")
print("  tail falls far slower than the Gaussian copula (same marginals AND correlation) and the")
print("  gap GROWS with severity (the rising ratio above). Tail-independence cannot represent")
print("  all-or-nothing contagion -- the 2008 lesson, from pure mechanics.\n")

# --- tail-sensitive BL overlay on the continuous returns ---
M = 1 << 16
seeds = default_rng(2).standard_normal((M, n)); L = np.linalg.cholesky(Ccont + 1e-6*np.eye(n))
center = lambda t: t - t.mean()
def W_gauss(theta): return np.bincount(np.argmin(theta + seeds @ L.T, 1), minlength=n) / M
def W_bowl(theta):
    idx = default_rng(3).integers(0, T, M)
    return np.bincount(np.argmin(theta + Rstd[idx], 1), minlength=n) / M
w0 = np.full(n, 1/n); theta = np.zeros(n)
for _ in range(15):
    F = W_gauss(theta) - w0
    if np.abs(F).sum() < 0.012: break
    J = np.column_stack([(W_gauss(theta+0.05*np.eye(n)[j]) - W_gauss(theta-0.05*np.eye(n)[j]))/0.10 for j in range(n)])
    theta = center(theta - np.linalg.lstsq(J, F, rcond=None)[0])
wg, wb = W_gauss(theta), W_bowl(theta); d = wb - wg; order = np.argsort(d)
print(f"benchmark reproduced under the correlation: |W_gauss - equal|_1 = {np.abs(wg-w0).sum():.3f}")
print("overlay shades the contagion-prone firms (high joint-default, dense neighbourhoods):")
for i in order[:5]:
    print(f"    firm {i:2d}  default {pmarg[i]:.0%}  tilt {d[i]:+.4f}")

def figure(path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    s = ax.scatter(X0, -Y0, c=d*1e3, cmap="RdYlGn", s=120+pmarg*900, edgecolors="k", linewidths=0.5,
                   vmin=-np.abs(d).max()*1e3, vmax=np.abs(d).max()*1e3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Bowling as a generative model for assets\n"
                 "random setup; marker size = default rate, colour = overlay tilt\n"
                 "(red = shaded for contagious co-default)")
    fig.colorbar(s, ax=ax, shrink=0.7, label="tail tilt (per mille)")
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); print(f"\nfigure -> {path}")

import os
figure(os.path.join(os.path.dirname(__file__), "bowling_assets.png"))
