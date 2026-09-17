"""Generate the embedded data for the tail-consistency demo.

Twelve assets: cluster A (5 names, Gaussian equicorrelated), cluster B (5
names, Clayton copula with standard-normal margins -- they crash together),
and two independent singles. At every slider position cluster A's Gaussian
correlation is set to the measured normal-scores correlation of cluster B's
Clayton copula, so the two clusters have the SAME correlation matrix (and unit
marginals) -- any covariance functional treats them identically by
construction. Only the joint lower tail differs.

Per lambda_L on the grid the script emits:
  wTrue : per-name weights of the loss-side race (argmin of centered
          performance) under the true simulation -- the tail-aware allocation
  wGauss: per-name weights of the Gaussian race driven by the empirical
          correlation of the same simulation (transport_weights) -- tail-blind
  wMV   : long-only-normalized minimum variance on the empirical covariance
  rho   : the matched within-cluster correlation
  dev   : max |corr_A - corr_B| entry deviation (verification of "identical")
  crashB, crashA : P(all five names in the cluster < -1 on the same day)
"""
import json, os, numpy as np
from numpy.random import default_rng
from scipy.stats import norm
from allocation._thurstone.covariance import nearest_correlation
from allocation._thurstone.transport import transport_weights

M = 400_000
NA, NB, NS = 5, 5, 2
N = NA + NB + NS
THETAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

rng = default_rng(0)
seeds = default_rng(1).standard_normal((1 << 16, N))


def clayton_block(theta, n, m):
    V = rng.gamma(1.0 / theta, 1.0, size=(m, 1))
    E = rng.exponential(1.0, size=(m, n))
    U = (1.0 + E / V) ** (-1.0 / theta)
    return norm.ppf(np.clip(U, 1e-12, 1 - 1e-12))


def gaussian_block(rho, n, m):
    Z = rng.standard_normal((m, 1))
    eps = rng.standard_normal((m, n))
    return np.sqrt(rho) * Z + np.sqrt(1.0 - rho) * eps


out = []
for theta in THETAS:
    lam = 2.0 ** (-1.0 / theta)
    B = clayton_block(theta, NB, M)
    CB = np.corrcoef(B.T)
    rho = float(CB[np.triu_indices(NB, 1)].mean())
    A = gaussian_block(rho, NA, M)
    S = rng.standard_normal((M, NS))
    X = np.hstack([A, B, S])
    Xc = X - X.mean(0)

    # true-simulation loss-side race
    idx = np.argmin(Xc, axis=1)
    wTrue = np.bincount(idx, minlength=N) / M

    # covariance-based views of the same data
    C = nearest_correlation(np.corrcoef(X.T))
    CA, CBm = C[:NA, :NA], C[NA:NA + NB, NA:NA + NB]
    dev = float(np.abs(CA - CBm).max())
    wGauss = np.asarray(transport_weights(np.zeros(N), C, seeds))
    mv = np.linalg.solve(np.cov(X.T), np.ones(N))
    mv = np.clip(mv, 0, None); mv = mv / mv.sum()

    crashA = float((X[:, :NA] < -1).all(axis=1).mean())
    crashB = float((X[:, NA:NA + NB] < -1).all(axis=1).mean())
    out.append({"lam": round(lam, 3), "rho": round(rho, 3), "dev": round(dev, 3),
                "wTrue": [round(float(v), 4) for v in wTrue],
                "wGauss": [round(float(v), 4) for v in wGauss],
                "wMV": [round(float(v), 4) for v in mv],
                "crashA": round(crashA, 5), "crashB": round(crashB, 5)})
    print(f"theta={theta:5.2f} lam={lam:.3f} rho={rho:.3f} dev={dev:.3f} | "
          f"race A={wTrue[:NA].sum():.3f} B={wTrue[NA:NA+NB].sum():.3f} | "
          f"gauss B={wGauss[NA:NA+NB].sum():.3f} mv B={mv[NA:NA+NB].sum():.3f} | "
          f"crash A={crashA*100:.2f}% B={crashB*100:.2f}%")

data = {"nA": NA, "nB": NB, "nS": NS, "points": out}
path = os.path.join(os.path.dirname(__file__), "..", "docs", "demos", "tail-consistency", "data.js")
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    f.write("const TC = " + json.dumps(data, separators=(",", ":")) + ";\n")
print(f"wrote {os.path.relpath(path)}  ({os.path.getsize(path)} bytes)")
