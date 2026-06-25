"""MoE router tilt: dependence-aware gating that de-duplicates redundant experts.

A Mixture-of-Experts gate maps a token to a distribution over experts (softmax of
affinities, then top-k). Two known pains: (i) softmax routing is correlation-blind
(IIA == independent latent utilities), so near-duplicate / collapsed experts each get
full gate weight -- the redundant cluster is double-counted; (ii) hard top-k routing
churns -- the chosen expert flips under small drift ('routing fluctuation').

The race tilt replaces the softmax with a noisy race whose latent utilities are
CORRELATED by expert similarity (C = Gram matrix of unit expert keys). Redundant
experts then compete with each other and SHARE gate mass (clone consistency), while
the gate stays a smooth function of the token (low churn). We isolate the routing
mechanism (no training): E experts, two of them near-duplicates.

Convention: package race is min-wins, so ability = -affinity (high affinity wins).
"""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from numpy.random import default_rng
from allocation._thurstone.transport import transport_weights

rng = default_rng(0)
d, E, M, scale = 6, 5, 1 << 14, 2.5
K = np.zeros((E, d))                                  # expert key vectors
K[0] = np.eye(d)[0] + 0.05 * rng.standard_normal(d)   # experts 0,1 are near-duplicates
K[1] = np.eye(d)[0] + 0.05 * rng.standard_normal(d)
K[2], K[3], K[4] = np.eye(d)[1], np.eye(d)[2], np.eye(d)[3]   # 2,3,4 distinct
K /= np.linalg.norm(K, axis=1, keepdims=True)
C = K @ K.T; np.fill_diagonal(C, 1.0)                 # similarity = valid correlation (Gram of unit keys)
print(f"expert-similarity (cos): dup pair (0,1)={C[0,1]:.3f}, distinct (0,2)={C[0,2]:.3f}\n")

def softmax(t): e = np.exp(t - t.max()); return e / e.sum()
def race(theta, Cmat, seeds): return np.asarray(transport_weights(-theta, Cmat, seeds))

# --- (1) clone-consistency: a token equally aligned with the duplicated direction
#         and a distinct expert. Fair gate -> dup PAIR gets ~ one distinct expert's mass.
x = K[0] + K[2]; x /= np.linalg.norm(x)
theta = scale * (K @ x)
seeds = default_rng(1).standard_normal((M, E))
print("=== (1) clone-consistency (token aligned equally with dup-dir and expert 2) ===")
print(f"affinities: {np.round(theta, 2)}  (experts 0,1,2 tie; 3,4 low)")
print(f"{'router':18}{'gate weights':>34}{'(w0+w1)/w2':>13}")
for name, w in [("softmax", softmax(theta)),
                ("race  C=I (IIA)", race(theta, np.eye(E), seeds)),
                ("race  C=keys", race(theta, C, seeds))]:
    print(f"{name:18}{np.array2string(w, precision=3):>34}{(w[0]+w[1])/w[2]:>13.2f}")
print("  fair value is ~1 (the dup pair is one effective expert at the same affinity as 2);")
print("  softmax and the independent race double-count it (~2); the key-correlated race de-dups.\n")

# --- (2) smoothness: drift the token so the top expert switches 2->3; measure turnover
print("=== (2) smoothness under drift (top expert switches 2 -> 3) ===")
path = [(1 - a) * K[2] + a * K[3] for a in np.linspace(0, 1, 21)]
path = [v / np.linalg.norm(v) for v in path]
prev = {}; tv = {"hard top-1": 0.0, "softmax": 0.0, "race C=keys": 0.0}
for v in path:
    th = scale * (K @ v)
    cur = {"hard top-1": np.eye(E)[np.argmax(th)], "softmax": softmax(th),
           "race C=keys": race(th, C, seeds)}                # common seeds -> smooth transport
    for k in tv:
        if k in prev: tv[k] += np.abs(cur[k] - prev[k]).sum()
    prev = cur
for k in tv:
    print(f"  {k:14} total routing turnover along drift = {tv[k]:.2f}")
print("\nread: the distinctive win is CLONE-CONSISTENCY. The key-correlated race is the only")
print("router that de-duplicates the redundant pair -- cluster ratio 1.11 vs ~2.08 for softmax")
print("and the IIA race -- routing the pair to about one expert's worth and freeing the second")
print("slot; within the pair it concentrates on the marginally better duplicate, which is what")
print("MoE wants (use one slot, not waste two). On smoothness the race is CONTINUOUS (turnover")
print("1.79, below hard top-1's one-hot jump at 2.0) but it is NOT smoother than softmax (1.42)")
print("-- it is more decisive, nearer to top-k routing, and carries finite-M noise. So: race vs")
print("softmax buys clone-consistency (softmax is smooth but correlation-blind); race vs hard")
print("top-1 buys continuity. Load-balancing equalizes expert COUNTS but still treats duplicates")
print("as separate; the race de-dups in the gate. DeepSeek-V3's out-of-gradient bias is a")
print("natural drop-in point.")
