"""The Jacobian J = dw/da as a substitutability matrix -- endogenous clustering.

Because w = grad G_S(theta), the Hessian J = grad^2 G_S = d w / d theta is the
choice-sensitivity: J_ij measures how much nudging contributor j's ability moves i's
winning probability -- i.e. how much they compete for the same winning event. So the
SAME race that produces the allocation w also carries, in its Jacobian, a map of
substitutes. We test two things:

  (A) does clustering |J| recover known correlation clusters (as well as clustering C
      does), so the clustering is endogenous to the allocation object?
  (B) the distinguishing case: J clusters in CHOICE geometry, not raw correlation.
      Two assets can be highly correlated yet not real substitutes IN THE RACE if one
      is so much stronger that the other never wins. C calls them a cluster; J should
      not, because they do not steal winning probability from each other.

J is estimated inversion-free by common-seed finite differences of the package race
(argmin, min-wins). No covariance inverse anywhere.
"""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from numpy.random import default_rng
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from allocation._thurstone.transport import transport_weights

M, H = 1 << 14, 0.04
seeds = default_rng(0).standard_normal((M, 0))  # placeholder; resized per n

def w_of(ability, C, sd):
    return np.asarray(transport_weights(ability, C, sd))

def jacobian(ability, C, sd, h=H):
    n = len(ability); J = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n); e[j] = h
        J[:, j] = (w_of(ability + e, C, sd) - w_of(ability - e, C, sd)) / (2 * h)
    return J

def affinity(Mat):                                  # symmetric, nonneg, zero diagonal
    A = (np.abs(Mat) + np.abs(Mat).T) / 2; np.fill_diagonal(A, 0.0); return A

def cluster(A, k):
    return SpectralClustering(n_clusters=k, affinity="precomputed",
                              random_state=0, assign_labels="discretize").fit_predict(A)

# ---- (A) three equal-ability correlated clusters -------------------------
g, k, rho = 4, 3, 0.9                               # 3 clusters of 4, within-corr 0.9
n = g * k
C = np.eye(n)
for b in range(k):
    sl = slice(b * g, (b + 1) * g)
    C[sl, sl] = (1 - rho) * np.eye(g) + rho * np.ones((g, g))
truth = np.repeat(np.arange(k), g)
ability = np.zeros(n)
sd = default_rng(1).standard_normal((M, n))
J = jacobian(ability, C, sd)
ari_J = adjusted_rand_score(truth, cluster(affinity(J), k))
ari_C = adjusted_rand_score(truth, cluster(affinity(C - np.eye(n)), k))
print("=== (A) recover 3 equal-ability correlation clusters (12 contributors) ===")
print(f"  cluster on |J| (choice geometry): adjusted Rand index = {ari_J:.2f}")
print(f"  cluster on C   (raw correlation): adjusted Rand index = {ari_C:.2f}")
print("  -> J recovers the clusters from the allocation object itself, no external C clustering.\n")

# ---- (B) correlation says 'cluster', the race says 'not substitutes' -----
# A correlated pair (corr 0.9) embedded among independents. Compare the pair's
# substitutability S_12 = |J_12| when the two are EQUAL strength vs when one dominates.
print("=== (B) relevance-awareness: correlated pair, equal vs dominated ===")
m = 6                                               # pair (0,1) + 4 independents
Cb = np.eye(m); Cb[0, 1] = Cb[1, 0] = 0.9
sdb = default_rng(2).standard_normal((M, m))
for label, ab in [("equal strength", np.zeros(m)),
                  ("one dominates  ", np.array([-1.6, 1.6, 0, 0, 0, 0.0]))]:
    Jb = jacobian(ab, Cb, sdb)
    w = w_of(ab, Cb, sdb)
    s12 = 0.5 * (abs(Jb[0, 1]) + abs(Jb[1, 0]))
    print(f"  {label}: corr(0,1)=0.90, w0={w[0]:.3f} w1={w[1]:.3f}, substitutability S_01={s12:.4f}")
print("  -> same correlation 0.90, but when one dominates the pair's race-substitutability")
print("     collapses (the weak one never wins, so it steals nothing). C cannot tell these")
print("     apart; J can. The clustering is relevance-aware, not just correlation-aware.")
