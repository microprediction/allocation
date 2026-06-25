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
