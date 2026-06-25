"""Bowling laboratory: a fully-known generative market for testing portfolios.

A rigid-body bowling simulation read as a market (Cotton): a pin is a firm, its position
diffuses under collisions (continuous stock movement), the knock-down threshold is a
DEFAULT barrier, and the chain reaction is DEFAULT CONTAGION. The setup is randomized into
tight clusters with gaps, so most throws miss (all quiet) and a throw into a cluster
cascades it whole -- bimodal, all-or-nothing co-movement a Gaussian copula cannot represent.

This module is the reusable lab: `generate()` returns a return panel with known,
controllable tail contagion, on which any allocator can be stress-tested out-of-sample.
Physics ported from humpday's bowling.html.
"""
import numpy as np
from numpy.random import default_rng

W = 800.0
MB, MP, E_BP, E_PP, PF, BFX, BFY = 10.0, 1.0, 0.5, 0.5, 0.96, 0.991, 0.991
BR, PR, BARRIER = 16.0, 9.0, 55.0          # ball radius, pin radius, default barrier (px)

def random_setup(rng, n=30, k=3, sd=5.0):
    """A random market: n firms in k tight clusters with gaps. Tighter sd => sharper
    all-or-nothing contagion. Randomizing gives a different market each call."""
    cx, cy = rng.uniform(300, 500, k), rng.uniform(200, 330, k)
    per = [n // k + (1 if i < n % k else 0) for i in range(k)]
    X0 = np.concatenate([cx[i] + rng.normal(0, sd, per[i]) for i in range(k)])
    Y0 = np.concatenate([cy[i] + rng.normal(0, sd, per[i]) for i in range(k)])
    return X0, Y0

def decode(u):                              # wide aim so most throws graze, cascades rare
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
        for i in range(n):                  # ball-pin collisions
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
        for i in range(n):                  # pin-pin collisions (the contagion)
            aMov = vx[i]*vx[i] + vy[i]*vy[i] > 1e-4
            for kk in range(i+1, n):
                if not aMov and not (vx[kk]*vx[kk] + vy[kk]*vy[kk] > 1e-4): continue
                dx, dy = x[kk]-x[i], y[kk]-y[i]; d2 = dx*dx + dy*dy; md = 2*PR
                if d2 >= md*md or d2 < 1e-6: continue
                d = np.sqrt(d2); nx, ny = dx/d, dy/d
                vrel = (vx[kk]-vx[i])*nx + (vy[kk]-vy[i])*ny
                if vrel > 0: continue
                j = -(1+E_PP)*vrel / (2/MP)
                vx[i] -= j*nx/MP; vy[i] -= j*ny/MP; vx[kk] += j*nx/MP; vy[kk] += j*ny/MP
                ov = md - d + 0.5
                x[i] -= nx*ov*0.5; y[i] -= ny*ov*0.5; x[kk] += nx*ov*0.5; y[kk] += ny*ov*0.5
    return np.hypot(x - X0, y - Y0)

def generate(n=30, T=2500, seed=7, k=3, sd=5.0, carry=0.10, scale=120.0):
    """Generate a bowling market. Returns (R, default, disp, X0, Y0):
      R       : T x n returns  (carry - displacement/scale): mostly small +, fat lower tail;
      default : T x n bool      (displacement past the barrier = jump-to-default);
      disp    : T x n raw pin displacements.
    """
    rng = default_rng(seed)
    X0, Y0 = random_setup(rng, n, k, sd)
    disp = np.array([throw(X0, Y0, rng.random(4)) for _ in range(T)])
    R = carry - disp / scale + 0.01 * rng.standard_normal((T, n))
    return R, disp > BARRIER, disp, X0, Y0

def tail_summary(R, default, ks=(3, 5, 8, 12), n_copula=300_000, seed=11):
    """Characterize a market's tail: correlation of returns + the joint-default tail ratios
    that DEVIATE from a Gaussian copula matched to the same marginals AND correlation.
    Returns (corr, marginal_default_rates, ratios{k:(bowling, copula)}, p_zero(bowling,copula))."""
    from scipy.stats import norm
    n = R.shape[1]
    C = np.corrcoef(((R - R.mean(0)) / R.std(0)).T)
    pmarg = default.mean(0)
    Z = default_rng(seed).multivariate_normal(np.zeros(n), C, size=n_copula)
    gdef = Z < norm.ppf(np.clip(pmarg, 1e-4, 1 - 1e-4))
    nb, ng = default.sum(1), gdef.sum(1)
    ratios = {k: (float(np.mean(nb >= k)), float(np.mean(ng >= k))) for k in ks}
    return C, pmarg, ratios, (float(np.mean(nb == 0)), float(np.mean(ng == 0)))

def fast(n=30, T=1000, seed=7, k=3, sd=5.0):
    """FAST MODE: ~T sims -> correlation matrix + Gaussian-copula tail-deviation ratios."""
    R, default, disp, X0, Y0 = generate(n=n, T=T, seed=seed, k=k, sd=sd)
    C, pmarg, ratios, pz = tail_summary(R, default)
    return dict(R=R, default=default, disp=disp, corr=C, pmarg=pmarg,
                tail_ratios=ratios, p_zero=pz, X0=X0, Y0=Y0)

if __name__ == "__main__":
    import sys
    n_, T_ = 30, (int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
    s = fast(n=n_, T=T_)
    off = s["corr"][~np.eye(n_, dtype=bool)]
    print(f"FAST MODE: {n_} firms, {T_} sims")
    print(f"correlation (off-diagonal): mean {off.mean():+.3f}, max {off.max():.3f}  "
          f"(full matrix in result['corr'])")
    print(f"marginal default rate: mean {s['pmarg'].mean():.1%}, max {s['pmarg'].max():.1%}")
    print(f"\n  P(0 defaults)  bowling {s['p_zero'][0]:.3f}  Gauss copula {s['p_zero'][1]:.3f}")
    print(f"  {'k+ default together':22}{'bowling':>10}{'copula':>10}{'ratio':>8}")
    for kk, (e, g) in s["tail_ratios"].items():
        print(f"  >= {kk:<3}{'':14}{e:>10.4f}{g:>10.4f}{(f'{e/g:.1f}x' if g else '>>'):>8}")
    print("\n  (ratio > 1 in the deep tail = contagion the Gaussian copula cannot represent)")
