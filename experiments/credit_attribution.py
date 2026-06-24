"""Redundancy-aware credit attribution: the Thurstone race vs the alternatives.

The race assigns each contributor its probability of *winning* a noisy contest --
P(i is the best). Read as credit attribution, this is exactly what you want when
contributors are correlated or redundant: credit for a shared success should be
*split*, not duplicated. This is the portfolio duplication paradox and the
discrete-choice red-bus/blue-bus (IIA) failure, in one mechanism.

Scenario 1 (red-bus/blue-bus). Three equally good contributors: A is independent;
B and B' are a correlated pair (correlation rho). As rho grows, B and B' become
the same horse, so a fair scheme must keep the (B,B') *cluster* credit at ~1/2 and
give A ~1/2. We compare:
  thurstone   : P(i = argmax performance)            -- the race / win frequency
  shapley     : Shapley value of the expected-max game v(S)=E[max_{i in S} X_i]
                (the gold-standard fair credit rule; redundancy-aware by axiom)
  iia         : credit under the independence assumption (correlation-blind)
The headline: thurstone tracks shapley (both conserve cluster credit); iia inflates
the duplicated cluster -- it robs A.

Scenario 2 (regression attribution is unstable under duplication). Contributors
are forecasters; credit = who is closest to the truth. Add a near-duplicate
forecaster and re-fit across noise seeds. OLS/NNLS credit flips arbitrarily
between the twins (the collinearity pathology); the race splits them evenly and
stably.
"""
import math, itertools, warnings, numpy as np
warnings.filterwarnings("ignore")
from numpy.linalg import lstsq
from scipy.optimize import nnls

rng = np.random.default_rng(0)
N = 400_000


# ----- credit rules -------------------------------------------------------
def draw(C, ability, n, seed):
    L = np.linalg.cholesky(C)
    Z = np.random.default_rng(seed).standard_normal((n, C.shape[0]))
    return ability + Z @ L.T                                  # (n, K) performances


def thurstone_credit(X):                                      # P(i = argmax) -- the race core
    return np.bincount(np.argmax(X, 1), minlength=X.shape[1]) / len(X)


def shapley_credit(X):                                        # Shapley of v(S)=E[max_S X]
    K = X.shape[1]
    vmax = lambda S: float(X[:, list(S)].max(1).mean()) if S else 0.0
    phi = np.zeros(K)
    for i in range(K):
        others = [j for j in range(K) if j != i]
        for r in range(len(others) + 1):
            wgt = math.factorial(r) * math.factorial(K - r - 1) / math.factorial(K)
            for S in itertools.combinations(others, r):
                phi[i] += wgt * (vmax(S + (i,)) - vmax(S))
    return phi / phi.sum()                                    # marginals >=0 -> shares


def iia_credit(C, ability, n, seed):                          # correlation-blind (independence)
    return thurstone_credit(draw(np.eye(C.shape[0]), ability, n, seed))


def markowitz_credit(C):                                      # population min-variance C^-1 1
    w = np.linalg.solve(C, np.ones(C.shape[0]))
    return w / w.sum()


# ----- Scenario 1: red-bus / blue-bus ------------------------------------
print("=== Scenario 1: red-bus/blue-bus  (A independent; B,B' correlated at rho) ===")
print("    three equally good contributors; fair credit keeps the (B,B') cluster at ~1/2\n")
print(f"{'rho':>5} | {'method':10} | {'A':>6} {'B':>6} {'Bdup':>6} | {'A':>6} {'B+Bdup cluster':>15}")
print("-" * 70)
rhos = [0.0, 0.3, 0.6, 0.9, 0.99]
S1 = {}
for rho in rhos:
    C = np.array([[1, 0, 0], [0, 1, rho], [0, rho, 1.0]])
    ability = np.zeros(3)
    creds = {"thurstone": thurstone_credit(draw(C, ability, N, 1)),
             "shapley":   shapley_credit(draw(C, ability, N, 1)),
             "markowitz": markowitz_credit(C),
             "iia":       iia_credit(C, ability, N, 1)}
    S1[rho] = creds
    for m, c in creds.items():
        print(f"{rho:>5.2f} | {m:10} | {c[0]:>6.3f} {c[1]:>6.3f} {c[2]:>6.3f} | "
              f"{c[0]:>6.3f} {c[1] + c[2]:>12.3f}")
    print()


# ----- Scenario 2: regression attribution instability under duplication ---
print("\n=== Scenario 2: forecast credit, add a near-duplicate forecaster ===")
print("    credit for the duplicated pair across 40 noise seeds: mean +/- sd of each twin\n")
T, base_corr = 300, 0.6
def forecast_credit_run(seed):
    g = np.random.default_rng(seed)
    y = g.standard_normal(T)
    # three genuinely distinct forecasters (correlated errors), then a duplicate of #2
    E = g.multivariate_normal(np.zeros(3),
            (1 - base_corr) * np.eye(3) + base_corr * np.ones((3, 3)), size=T) * 0.7
    dup = E[:, [1]] + g.standard_normal((T, 1)) * 0.05            # near-duplicate of forecaster 1 (B)
    E = np.hstack([E, dup])                                       # columns: A0, B, C, B'(dup)
    F = y[:, None] - E                                           # forecasts
    # race credit: who is closest to truth each period (argmin |error|) -- redundancy-aware
    th = np.bincount(np.argmin(np.abs(E), 1), minlength=4) / T
    # OLS / NNLS attribution: regress truth on forecasts, normalize |coef| to shares
    ols = np.abs(lstsq(F, y, rcond=None)[0]); ols = ols / ols.sum()
    nn, _ = nnls(F, y); nn = nn / nn.sum() if nn.sum() > 0 else np.full(4, .25)
    # Markowitz / Bates-Granger: min-variance on the estimated error covariance (Se^-1 1)
    wbg = np.linalg.pinv(np.cov(E.T)) @ np.ones(4); mk = wbg / wbg.sum()
    return th, ols, nn, mk

labels = ["A", "B", "C", "B'(dup)"]
runs = [forecast_credit_run(s) for s in range(40)]
TH = np.array([r[0] for r in runs]); OLS = np.array([r[1] for r in runs])
NN = np.array([r[2] for r in runs]); MK = np.array([r[3] for r in runs])
print(f"{'method':10} | " + "  ".join(f"{l:>11}" for l in labels) + "   | twin-split sd")
print("-" * 78)
for name, A in [("thurstone", TH), ("markowitz", MK), ("ols", OLS), ("nnls", NN)]:
    cells = "  ".join(f"{A[:, j].mean():>5.3f}±{A[:, j].std():>4.2f}" for j in range(4))
    twin_sd = float(A[:, 1].std() + A[:, 3].std())            # instability of the duplicated pair
    print(f"{name:10} | {cells}   | {twin_sd:.3f}")
print("\n(B and B'(dup) are the same forecaster twice: a fair scheme splits them evenly"
      "\n and stably. OLS/NNLS thrash between the twins across seeds; the race does not.)")


# ----- figure -------------------------------------------------------------
def figure(path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    # left: (B,B') CLUSTER credit vs rho -- fair schemes shed it toward 1/2, IIA does not
    cols = {"thurstone": "#2ca02c", "shapley": "#9467bd", "markowitz": "#1f77b4", "iia": "#d62728"}
    for m, col in cols.items():
        ax[0].plot(rhos, [S1[r][m][1] + S1[r][m][2] for r in rhos], "-o", ms=4, color=col, label=m)
    ax[0].axhline(0.5, color="k", lw=0.6, ls=":")
    ax[0].annotate("two-body 1/2\n(one horse)", (0.99, 0.5), (0.6, 0.55), fontsize=7,
                   arrowprops=dict(arrowstyle="->", lw=0.6))
    ax[0].set_xlabel("correlation rho of the duplicated pair (B,B')")
    ax[0].set_ylabel("credit to the (B,B') cluster")
    ax[0].set_title("Red-bus/blue-bus: cluster credit vs redundancy (population)")
    ax[0].legend(fontsize=8, frameon=False); ax[0].grid(alpha=0.3)
    # right: twin instability under estimation noise (scatter of B vs B' across seeds)
    for name, A, col in [("thurstone", TH, "#2ca02c"), ("markowitz", MK, "#1f77b4"),
                         ("nnls", NN, "#ff7f0e"), ("ols", OLS, "#8c564b")]:
        ax[1].scatter(A[:, 1], A[:, 3], s=14, color=col, alpha=0.6, label=name)
    ax[1].plot([-.2, .6], [-.2, .6], "k:", lw=0.6)
    ax[1].set_xlabel("credit to B"); ax[1].set_ylabel("credit to B'(duplicate)")
    ax[1].set_title("Duplication stability under estimation noise: B vs B' across seeds")
    ax[1].legend(fontsize=8, frameon=False); ax[1].grid(alpha=0.3)
    fig.suptitle("Thurstone race as redundancy-aware credit attribution", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"\nfigure -> {path}")

import os
figure(os.path.join(os.path.dirname(__file__), "credit_attribution.png"))
