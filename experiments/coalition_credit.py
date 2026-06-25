"""Does the race credit capture coalitions? No -- and here is the gap, concretely.

The race credit w_i = P(i is the single best) is the gradient of a winner-take-all
value E[max]. It is SELECTION relevance, not coalitional value: there is no notion of
subsets cooperating. Shapley of a COMBINATION-value game does capture cooperation:
v(S) = quality of the best blend of the contributors in S. A contributor that is
never individually best but diversifies every blend it joins should get large
coalitional credit and ~zero race credit.

Forecast-combination setup: y plus four forecasters.
  A, B : strong but redundant (error corr 0.9)         -- often individually best
  C    : a DIVERSIFIER, individually worse (higher var) but uncorrelated errors
  D    : weak, uncorrelated                              -- noise
We compare, as credit shares:
  race      : frequency forecaster i is closest to y per instance (argmin |error|)
  Shapley   : Shapley value of v(S) = R^2 of the OLS combination of {f_k : k in S}
The point: C should rank high under Shapley (it carries combination value) and low
under the race (it is rarely the single closest).
"""
import itertools, math, warnings, numpy as np
warnings.filterwarnings("ignore")
from numpy.random import default_rng

rng = default_rng(0); T = 4000
y = rng.standard_normal(T)
common = rng.standard_normal(T)
eA = np.sqrt(0.9) * common + np.sqrt(0.1) * rng.standard_normal(T)   # var 1
eB = np.sqrt(0.9) * common + np.sqrt(0.1) * rng.standard_normal(T)   # var 1, corr(A,B)=0.9
eC = np.sqrt(2.0) * rng.standard_normal(T)                           # var 2, uncorrelated (diversifier)
eD = np.sqrt(4.0) * rng.standard_normal(T)                           # var 4, uncorrelated (weak)
E = np.column_stack([eA, eB, eC, eD]); names = ["A", "B", "C", "D"]
F = y[:, None] + E                                                   # forecasts
n = 4

def r2(S):                                                          # OLS combination R^2 over coalition S
    if not S: return 0.0
    Xs = np.column_stack([np.ones(T)] + [F[:, j] for j in S])
    beta, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    res = y - Xs @ beta
    return float(1 - res @ res / (y @ y - y.sum() ** 2 / T))

def shapley():
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for r in range(len(others) + 1):
            c = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for S in itertools.combinations(others, r):
                phi[i] += c * (r2(list(S) + [i]) - r2(list(S)))
    return phi

phi = shapley(); phi_share = phi / phi.sum()
race = np.bincount(np.argmin(np.abs(E), 1), minlength=n) / T         # individually-closest frequency
indiv_r2 = np.array([r2([i]) for i in range(n)])                     # standalone accuracy

print(f"individual R^2:  " + "  ".join(f"{names[i]} {indiv_r2[i]:.2f}" for i in range(n)))
print(f"full-blend R^2 = {r2(list(range(n))):.3f}\n")
print(f"{'forecaster':12}{'race credit':>13}{'Shapley credit':>16}")
print("-" * 41)
for i in range(n):
    print(f"{names[i]:12}{race[i]:>13.3f}{phi_share[i]:>16.3f}")
print("\nread (honest): here the two credits roughly AGREE (rank A=B>C>D, close shares) --")
print("for moderate redundancy/diversification, selection credit is a decent proxy for")
print("coalitional value. The conceptual boundary is still real and the gap WIDENS for a")
print("purely-coalitional contributor (individually useless but combination-essential):")
print("the race, being the gradient of a winner-take-all E[max], can never credit a")
print("contributor that is never the single best, however much it improves blends. So the")
print("race is selection credit; true coalitional credit needs a combination-value game")
print("v(S)=quality of the best blend of S and its Shapley. They answer different questions,")
print("and coincide only when being-best-alone tracks adding-value-in-combination.")
