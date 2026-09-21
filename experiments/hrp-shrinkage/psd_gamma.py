"""The gamma at which the bridge's implied covariance stops being a fake.

Below it the nearest covariance reproducing the portfolio is indefinite: no
set of beliefs about risk, however odd, produces those weights as a minimum
variance solution without travelling further than the projection. Above it
the allocator's implied beliefs are a real covariance matrix.
"""
import numpy as np
from confound import true_sigma
from implied import implied
from allocation._schur.bridge import bridge_weights, bisection_tree
from allocation._schur.seriation import seriate

rng = np.random.default_rng(23)
grid = np.linspace(0.0, 1.0, 51)
cross, best = [], []
for _ in range(200):
    Sig, _ = true_sigma(rng, 5, 8, 0.7, 0.15)
    X = rng.normal(size=(120, 40)) @ np.linalg.cholesky(Sig).T
    S = np.cov(X, rowvar=False)
    tree = bisection_tree(seriate(S)[0], leaf_size=1)
    ratios, vars_ = [], []
    for g in grid:
        w = bridge_weights(S, tree, gamma=float(g), eta=1.0, split="dial")
        _, c = implied(S, w)
        ratios.append(c); vars_.append(float(w @ Sig @ w))
    ratios = np.asarray(ratios)
    ok = np.where(ratios > 0)[0]
    cross.append(grid[ok[0]] if len(ok) else np.nan)
    best.append(grid[int(np.argmin(vars_))])
cross = np.asarray(cross, dtype=float); best = np.asarray(best)
q = np.nanpercentile(cross, [10, 25, 50, 75, 90])
print("gamma at which the implied covariance first becomes positive definite")
print(f"  deciles 10/25/50/75/90:  {q[0]:.2f}  {q[1]:.2f}  {q[2]:.2f}  {q[3]:.2f}  {q[4]:.2f}")
print(f"  never positive definite in {int(np.isnan(cross).sum())} of {len(cross)} markets")
print()
print("gamma minimising true variance (oracle choice)")
print(f"  deciles 10/25/50/75/90:  " + "  ".join(f"{v:.2f}" for v in np.percentile(best, [10,25,50,75,90])))
print(f"  correlation between the two gammas: {np.corrcoef(cross[~np.isnan(cross)], best[~np.isnan(cross)])[0,1]:+.3f}")
