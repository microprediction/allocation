"""The bridge, read as a path in implied-estimator space.

If the implied-estimator distortion falls monotonically from the HRP end to
zero at the minimum-variance end, then the dial is literally interpolating
between two beliefs about the covariance, and the bridge is the controlled
experiment: one covariance in, a family of implied covariances out.
"""
import numpy as np
from confound import true_sigma
from implied import implied
from allocation._schur.bridge import bridge_weights, bisection_tree
from allocation._schur.seriation import seriate

rng = np.random.default_rng(11)
GAMMAS = np.linspace(0.0, 1.0, 11)
dist = {g: [] for g in GAMMAS}
oos = {g: [] for g in GAMMAS}
psd = {g: [] for g in GAMMAS}

for _ in range(200):
    Sig, _ = true_sigma(rng, 5, 8, 0.7, 0.15)
    X = rng.normal(size=(120, 40)) @ np.linalg.cholesky(Sig).T
    S = np.cov(X, rowvar=False)
    tree = bisection_tree(seriate(S)[0], leaf_size=1)
    for g in GAMMAS:
        w = bridge_weights(S, tree, gamma=float(g), eta=1.0, split="dial")
        d, c = implied(S, w)
        dist[g].append(d); psd[g].append(c); oos[g].append(float(w @ Sig @ w))

print("Schur bridge, HRP at gamma=0 to minimum variance at gamma=1")
print(f"{'gamma':>6s} {'distortion':>12s} {'lmin/lmax':>11s} {'true variance':>15s}")
for g in GAMMAS:
    print(f"{g:6.1f} {np.median(dist[g]):12.3f} {np.median(psd[g]):11.4f} {np.median(oos[g]):15.5f}")
d0, d1 = np.median(dist[0.0]), np.median(dist[1.0])
print(f"\nmonotone in gamma: {all(np.median(dist[GAMMAS[i]]) >= np.median(dist[GAMMAS[i+1]]) - 1e-12 for i in range(len(GAMMAS)-1))}")
print(f"distortion at the two ends: {d0:.3f} -> {d1:.3g}")
