"""Does the win-probability bridge to a Shapley value? A numerical test.

The race's win-probability is p_i = P(i = argmax X), X_i = a_i + eps_i. The paper's
implied-objective identity says p = grad G, where G(a) = E[max_i(a_i + eps_i)] is the
expected maximum (a convex 'potential'). Shapley-type values are diagonal integrals
of marginals (Owen 1972: Shapley = integral over t of the multilinear extension's
gradient along the diagonal). So we ask, on a small game where exact Shapley is
computable:

  (1) does p = grad G hold?                          (gradient identity)
  (2) is the diagonal integral of the win-probability an efficient, symmetric,
      Shapley-TYPE value?                             (the Aumann-Shapley bridge)
  (3) does it equal the Shapley value of the expected-max MEMBERSHIP game
      v(S) = E[max_{i in S} X_i]?                     (is it a drop-in substitute?)

Game: 5 contributors. Players 3 and 4 (0-based) are an exchangeable near-duplicate
pair (equal ability, correlated 0.95); 0,1,2 are distinct and independent.
"""
import itertools, math, numpy as np

n, N, h = 5, 400_000, 1e-2
a = np.array([1.0, 0.6, 0.2, 0.4, 0.4])              # players 3,4 share ability 0.4
C = np.eye(n); C[3, 4] = C[4, 3] = 0.95              # ... and are a near-duplicate pair
L = np.linalg.cholesky(C)
Z = np.random.default_rng(0).standard_normal((N, n))  # common seeds -> smooth in a
E = Z @ L.T

def Xb(b):  return b + E                              # performances at ability vector b
def Gb(b):  return float(Xb(b).max(1).mean())         # expected maximum  G(b)
def winprob(b):
    return np.bincount(Xb(b).argmax(1), minlength=n) / N


# (1) gradient identity:  p_i =?= dG/da_i  (central difference)
p = winprob(a)
grad = np.array([(Gb(a + h * np.eye(n)[i]) - Gb(a - h * np.eye(n)[i])) / (2 * h)
                 for i in range(n)])
print("(1) gradient identity  p = grad G")
print(f"    win-prob p   : {np.array2string(p, precision=4)}")
print(f"    dG/da (fd)    : {np.array2string(grad, precision=4)}")
print(f"    max |p - grad|: {np.max(np.abs(p - grad)):.2e}\n")


# (2) Aumann-Shapley value of the ABILITY-scaling game f(t) = E[max(t*a_i + eps_i)].
#     AS_i = a_i * integral_0^1 p_i(t*a) dt   (win-prob is the gradient integrand)
ts = np.linspace(0.0, 1.0, 41)
Pt = np.array([winprob(t * a) for t in ts])          # (41, n) win-prob along the diagonal
AS = a * np.trapezoid(Pt, ts, axis=0)
total = Gb(a) - Gb(np.zeros(n))                       # what an efficient value must sum to
print("(2) Aumann-Shapley value of the ability game, built from win-probabilities")
print(f"    AS_i          : {np.array2string(AS, precision=4)}")
print(f"    sum AS_i      : {AS.sum():.4f}   target G(a)-G(0) = {total:.4f}   "
      f"(efficiency gap {abs(AS.sum() - total):.2e})")
print(f"    symmetry AS[3] vs AS[4]: {AS[3]:.4f} vs {AS[4]:.4f}   "
      f"(diff {abs(AS[3] - AS[4]):.2e})\n")


# (3) exact Shapley of the membership game v(S) = E[max_{i in S} X_i], v(empty)=0
X = Xb(a)
def v(S): return float(X[:, list(S)].max(1).mean()) if S else 0.0
phi = np.zeros(n)
for i in range(n):
    others = [j for j in range(n) if j != i]
    for r in range(len(others) + 1):
        c = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
        for S in itertools.combinations(others, r):
            phi[i] += c * (v(S + (i,)) - v(S))
phi_share = phi / phi.sum()
print("(3) exact Shapley of the expected-max membership game vs the win-probability")
print(f"    Shapley share : {np.array2string(phi_share, precision=4)}")
print(f"    win-prob p    : {np.array2string(p, precision=4)}")
print(f"    symmetry phi[3] vs phi[4]: {phi_share[3]:.4f} vs {phi_share[4]:.4f}")
print(f"    duplicate pair (3+4) total -- Shapley {phi_share[3]+phi_share[4]:.4f}  "
      f"vs win-prob {p[3]+p[4]:.4f}")
print(f"    max |Shapley share - win-prob|: {np.max(np.abs(phi_share - p)):.4f}")

print("\nverdict:")
print("  (1) holds: the win-probability IS the gradient of the expected-max potential.")
print("  (2) holds: its diagonal integral is an efficient, symmetric Aumann-Shapley")
print("      value -- of the ABILITY-scaling game. So win-probabilities are a bona fide")
print("      Shapley-type gradient integrand.")
print("  (3) the win-probability is NOT equal to the Shapley value of the MEMBERSHIP")
print("      game; both are symmetric, but they weight the duplicated pair differently")
print("      relative to the field (direction is configuration-dependent: here Shapley")
print("      gives the pair slightly more; in the equal-ability sweep it gave the cluster")
print("      less). Same expected-max object, different value -- a cousin, not a drop-in.")
