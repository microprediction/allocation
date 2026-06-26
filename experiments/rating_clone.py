"""Clone-aware rating: Bradley-Terry + a dependence tilt for near-duplicate competitors.

The Thurstone->Bradley-Terry->Elo lineage is this construction's native home. BT
ratings are fine PAIRWISE: two near-duplicate competitors (same base model, sibling
finetunes; same-stable horses) correctly get equal ratings. The flaw appears when a
leaderboard is turned into a FIELD quantity -- "probability model i is the best", or a
preference / selection share -- by the standard softmax/Luce of the ratings. That step
is IIA (independent latent performances), so a cluster of k near-duplicate top models
each gets full share and the cluster DOUBLE-COUNTS (red-bus/blue-bus): it collectively
looks ~k x as good as one distinct model of equal skill. Timely for LLM arena
leaderboards (Chatbot Arena / LMSYS use BT, with many near-duplicate models).

Fix: treat the ratings as abilities of a race whose latent performances are CORRELATED
by model similarity (estimated from the data: models with the same win-rate profile
against the field are near-duplicates), and read off the tilted field share. Clones
then compete with each other and share one model's worth.

Convention: package race is min-wins, so ability = -rating (higher rating wins).
"""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from numpy.random import default_rng
from sklearn.linear_model import LogisticRegression
from allocation._thurstone.transport import transport_weights

rng = default_rng(0)
n, R = 5, 6000                                            # 5 models, R rounds of all-pairs play
a = np.array([1.2, 1.2, 0.4, 0.4, -0.5])                 # models 0,1 are clones (equal top ability)
eps = rng.standard_normal((R, n))
eps[:, 1] = 0.97 * eps[:, 0] + np.sqrt(1 - 0.97 ** 2) * eps[:, 1]   # clone 1 co-moves with 0
X = a + eps                                              # latent per-round performances

# all-pairs outcomes each round -> Bradley-Terry fit + per-round win counts
feats, ys = [], []; prw = np.zeros((R, n))               # prw[r,i] = games i won in round r
for r in range(R):
    for i in range(n):
        for j in range(i + 1, n):
            f = np.zeros(n); f[i], f[j] = 1.0, -1.0
            iwin = int(X[r, i] > X[r, j]); feats.append(f); ys.append(iwin)
            prw[r, i] += iwin; prw[r, j] += 1 - iwin
ratings = LogisticRegression(fit_intercept=False, C=1e3).fit(np.array(feats), np.array(ys)).coef_[0]
ratings -= ratings.mean()
C = np.corrcoef(prw.T)                                   # similarity = co-movement of per-round wins
np.fill_diagonal(C, 1.0)
print(f"BT ratings: {np.round(ratings,2)}  (clones 0,1 equal -- pairwise BT is fine)")
print(f"data-derived similarity (per-round win co-movement): clones C[0,1]={C[0,1]:.3f}, "
      f"distinct C[0,2]={C[0,2]:.3f}\n")

def softmax(r): e = np.exp(r - r.max()); return e / e.sum()
def race(r, Cmat):
    seeds = default_rng(1).standard_normal((1 << 14, len(r)))
    return np.asarray(transport_weights(-r, Cmat, seeds))

# fair target: replace the clone pair by ONE model of the same rating -> its field share
keep = [0, 2, 3, 4]                                      # drop clone 1; model 0 represents the pair
fair = softmax(ratings[keep])[0]                         # a single top model's share in a 4-field

C_meta = np.eye(n); C_meta[0, 1] = C_meta[1, 0] = 0.95   # siblings flagged same base model
w_sm, w_ri = softmax(ratings), race(ratings, np.eye(n))
w_rc, w_meta = race(ratings, C), race(ratings, C_meta)
print(f"{'field share P(best)':28}{'clone cluster w0+w1':>22}{'vs fair single':>16}")
print("-" * 66)
for name, w in [("softmax (Luce / IIA)", w_sm), ("race  C=I (IIA)", w_ri),
                ("race  C=data (per-round)", w_rc), ("race  C=metadata (siblings)", w_meta)]:
    cl = w[0] + w[1]
    print(f"{name:28}{cl:>22.3f}{cl / fair:>15.2f}x")
print(f"\nfair single-model share (the exact-clone, C->1 limit) = {fair:.3f}")
print("\nread: BT ratings are correct (clones tie -- pairwise BT is fine). The field 'P(best)'")
print(f"share is where the IIA flaw bites: treating the 95%-correlated siblings as INDEPENDENT,")
print(f"the IIA race over-counts the pair at {(w_ri[0]+w_ri[1])/fair:.2f}x a single equal model. Re-running the SAME")
print(f"race under sibling correlation corrects it -- {(w_rc[0]+w_rc[1])/fair:.2f}x from a weak data-derived C (per-round")
print(f"win co-movement is a coarse proxy; siblings even play each other), {(w_meta[0]+w_meta[1])/fair:.2f}x from a metadata")
print("C that just flags same-base siblings. The residual above 1.00x is legitimate (95%-, not")
print("100%-correlated siblings keep a small two-shot edge); only exact clones collapse to fair.")
print("So: keep BT for pairwise skill; compute field shares / 'P(best)' / routing weights with a")
print("clone-aware race, the de-dup tracking how well similarity is estimated. (Softmax's lower")
print("number is the logistic link, not de-dup -- it is IIA too; the apples-to-apples de-dup is")
print("C=I -> C=metadata within the race.) Direct fit: BT-based LLM-arena leaderboards full of")
print("sibling models, where 'which is best / route to' double-counts a family and same-base")
print("metadata gives a strong C.")
