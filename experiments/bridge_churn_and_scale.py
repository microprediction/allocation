"""Two claims behind SchurBridge, checked numerically.

1. Reclustering churn is damped by the dials: with exact conditioning ('tree',
   'flat') the weight change caused by a membership change shrinks to zero as
   (gamma, eta) -> (1, 1). With 'knots' it does not on a generic covariance,
   because the one-factor approximation itself depends on the partition.
2. Timing of the three conditionings at a few sizes.
"""
import time
import numpy as np
from allocation._schur.bridge import bridge_weights, contiguous_partition

rng = np.random.default_rng(0)
n, k = 60, 6
G = rng.standard_normal((n, 4 * n)); cov = G @ G.T / (4 * n) + np.diag(rng.uniform(0.2, 0.6, n))
order = rng.permutation(n)
partA = contiguous_partition(order, k)
order2 = order.copy(); i = 9; order2[i], order2[i + 1] = order2[i + 1], order2[i]   # one crossing at a cut boundary
partB = contiguous_partition(order2, k)
print("weight change (L1) from one membership swap:")
for cond in ("tree", "flat", "knots"):
    row = []
    for g, e in [(0, 0), (0.5, 0.5), (0.9, 0.9), (1, 1)]:
        wa = bridge_weights(cov, partA, gamma=g, eta=e, conditioning=cond)
        wb = bridge_weights(cov, partB, gamma=g, eta=e, conditioning=cond)
        row.append(f"({g},{e}) {np.abs(wa - wb).sum():.4f}")
    print(f"  {cond:<6}", "  ".join(row))
print("  (exact conditionings reach zero at (1,1); knots only under a block one-factor model)")

print("\ntiming at scale (seconds per allocation):")
for n, k in [(300, 10), (1000, 20), (2000, 40)]:
    G = rng.standard_normal((n, 2 * n)); cov = G @ G.T / (2 * n) + np.diag(rng.uniform(0.2, 0.6, n))
    part = contiguous_partition(rng.permutation(n), k)
    for cond in ("knots", "tree"):
        t = time.perf_counter(); bridge_weights(cov, part, gamma=0.7, eta=1.0, conditioning=cond); dt = time.perf_counter() - t
        print(f"  n={n:<5} k={k:<3} {cond:<6} {dt:.2f}s")
