"""Behind the first table on schur.microprediction.org/where-to-damp.html; moved here from the schur repo.

Schur damping in the estimate versus in the algorithm, out of sample.
Requires the allocation package (pip install allocation). Reproduces the table on docs/where-to-damp.html."""
import numpy as np
from allocation._schur.bridge import bridge_weights
from allocation.convex import min_variance_weights
rng = np.random.default_rng(5)
n, K, M = 32, 4, 8
lab = np.repeat(np.arange(K), M); same = lab[:, None] == lab[None, :]
def market():
    F = np.eye(K); g = 0.45 + 0.3*rng.random(K)
    for a in range(K):
        for b in range(a): F[a, b] = F[b, a] = g[a]*g[b]*(0.5 + 0.3*rng.random())
    vol = 0.12 + 0.28*rng.random(n); load = 0.35 + 0.4*rng.random(n); bm = 0.25*(0.6 + 0.8*rng.random(n))
    S = np.outer(load, load)*F[np.ix_(lab, lab)] + np.outer(bm, bm); S += np.diag(np.maximum(1 - np.diag(S), 0.15))
    return S*np.outer(vol, vol)
grid = np.linspace(0, 1, 11); T = 90; nm = 24
clusters = [np.where(lab == k)[0] for k in range(K)]
names = ('algorithm: flat bridge on the raw estimate', 'estimate: taper the cross-block by sqrt(g), then min-var',
         'estimate: taper the cross-block by g, then min-var', 'both: taper by sqrt(g), then the bridge at g')
acc = {k: np.zeros(len(grid)) for k in names}
for _ in range(nm):
    S = market(); X = rng.multivariate_normal(np.zeros(n), S, size=T); Sh = np.cov(X, rowvar=False)
    best = min_variance_weights(S); vb = best @ S @ best
    ex = lambda w: 100*((w @ S @ w)/vb - 1)
    for i, g in enumerate(grid):
        Ts = np.where(same, 1.0, np.sqrt(g)); Tg = np.where(same, 1.0, g)
        acc[names[0]][i] += ex(bridge_weights(Sh, clusters, gamma=g, eta=1.0, conditioning='all'))/nm
        acc[names[1]][i] += ex(min_variance_weights(Sh*Ts))/nm
        acc[names[2]][i] += ex(min_variance_weights(Sh*Tg))/nm
        acc[names[3]][i] += ex(bridge_weights(Sh*Ts, clusters, gamma=g, eta=1.0, conditioning='all'))/nm
print(f"excess variance over the true optimum (%), T={T}, {nm} markets; dial 0..1")
for k, v in acc.items(): print(f"  {k:56s}", " ".join(f"{x:5.1f}" for x in v), f"  min {v.min():.1f} at {grid[v.argmin()]:.1f}")
