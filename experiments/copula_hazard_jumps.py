"""The Gaussian copula of default TIMES has ugly dynamics: conditional hazards jump.

Sklar's theorem (the distributional/probability-integral transform) is a triviality; the
copula-of-default-times model staples it onto temporal quantities and then has no coherent
answer to "what happens after t>0?". The only honest move is to CONDITION on the information
revealed (which names have defaulted, when; which survive to t) -- but then the implied hazard
rates of the survivors JUMP, and you have left the model class you started in.

Mechanics. Latent X ~ N(0, Sigma), equicorrelation rho. Exponential marginals (hazard lam):
default time tau_i = -ln(1-U_i)/lam, U_i = Phi(X_i). Surviving to t  <=>  X_i > c(t),
c(t) = Phi^{-1}(F(t)), F(t)=1-e^{-lam t}. A default at tau_j pins X_j = c(tau_j) exactly.

Conditional hazard of a survivor i, given defaulters D (known latents) and all survivors S
alive to t (a RECTANGLE / orthant condition X_k > c(t)):
  h_i(t) = g_{i|D}(c(t)) * c'(t) * P(X_k>c(t), k in S\\{i} | X_i=c(t), D) / P(X_k>c(t), k in S | D)
with c'(t) = f(t)/phi(c(t)), f(t)=lam e^{-lam t}. Both probabilities are multivariate-normal
orthant probabilities -> computed here with scipy and CACHED to a data file for the web demo.
We report the hazard MULTIPLIER R_i(t) = h_i(t)/lam (1 = the standalone marginal hazard).
"""
import json, os, warnings, numpy as np
warnings.filterwarnings("ignore")
from numpy.random import default_rng
from scipy.stats import norm, multivariate_normal as MVN

n, lam, T = 5, 0.06, 5.0
names = ["A", "B", "C", "D", "E"]
F   = lambda t: 1 - np.exp(-lam * t)
Fi  = lambda u: -np.log(1 - u) / lam
cth = lambda t: norm.ppf(F(t))                 # survival threshold: X_i > cth(t) <=> tau_i > t

def gauss_cond(mean, cov, idx, vals):
    """Condition N(mean,cov) on coords idx==vals; return (mean,cov,rest_idx) for the rest."""
    idx = list(idx); rest = [k for k in range(len(mean)) if k not in idx]
    if not rest: return np.zeros(0), np.zeros((0, 0)), rest
    m, C = np.asarray(mean, float), np.asarray(cov, float)
    Coo = C[np.ix_(idx, idx)]; Cro = C[np.ix_(rest, idx)]; Crr = C[np.ix_(rest, rest)]
    K = Cro @ np.linalg.inv(Coo)
    return m[rest] + K @ (np.asarray(vals) - m[idx]), Crr - K @ Cro.T, rest

def orthant_upper(mean, cov, c):
    """P(Y > c) for Y ~ N(mean,cov)  ==  P(N(0,cov) < mean - c)."""
    d = len(mean)
    if d == 0: return 1.0
    if d == 1: return float(norm.sf((c[0] - mean[0]) / np.sqrt(cov[0, 0])))
    return float(MVN(mean=np.zeros(d), cov=cov, allow_singular=True).cdf(np.asarray(mean) - np.asarray(c)))

def hazard_paths(rho, seed):
    Sigma = (1 - rho) * np.eye(n) + rho * np.ones((n, n))
    L = np.linalg.cholesky(Sigma)
    rng = default_rng(seed)
    for _ in range(2000):                       # pick a scenario with 2-3 defaults before T
        X = L @ rng.standard_normal(n)
        tau = Fi(norm.cdf(X))
        if 2 <= int((tau < T).sum()) <= 3: break
    ts = np.linspace(0.02, T, 320)
    R = {nm: [] for nm in names}
    for t in ts:
        D = [j for j in range(n) if tau[j] <= t]        # defaulted (latent known = X[j])
        Sset = [i for i in range(n) if tau[i] > t]       # survivors (rectangle X>cth(t))
        if D: muS, covS, Sord = gauss_cond(np.zeros(n), Sigma, D, X[D])
        else: muS, covS, Sord = np.zeros(n), Sigma.copy(), list(range(n))
        c = cth(t)
        denom = orthant_upper(muS, covS, [c] * len(Sord))
        for nm_i, i in zip(names, range(n)):
            if i not in Sord:
                R[nm_i].append(None); continue
            li = Sord.index(i)
            mu2, cov2, _ = gauss_cond(muS, covS, [li], [c])
            num = orthant_upper(mu2, cov2, [c] * len(mu2))
            sig = np.sqrt(covS[li, li]); g = norm.pdf((c - muS[li]) / sig) / sig
            cprime = (lam * np.exp(-lam * t)) / norm.pdf(c)
            h = g * cprime * num / max(denom, 1e-12)
            R[nm_i].append(round(float(h / lam), 4))
    defaults = sorted([{"name": names[j], "time": round(float(tau[j]), 3)}
                       for j in range(n) if tau[j] < T], key=lambda d: d["time"])
    return {"rho": rho, "t": [round(float(x), 3) for x in ts], "R": R, "defaults": defaults}

worlds = [hazard_paths(rho, seed) for rho, seed in [(0.3, 4), (0.5, 7), (0.7, 11)]]
data = {"lam": lam, "T": T, "names": names, "worlds": worlds}
for w in worlds:
    print(f"rho={w['rho']}: defaults " + ", ".join(f"{d['name']}@{d['time']}" for d in w["defaults"]))
    jumps = []
    for d in w["defaults"]:
        ti = min(range(len(w["t"])), key=lambda k: abs(w["t"][k] - d["time"]))
        for nm in names:
            a = w["R"][nm][max(0, ti - 1)]; b = w["R"][nm][min(len(w["t"]) - 1, ti + 1)]
            if a and b and b / a > 1.3: jumps.append(f"{nm} x{a:.1f}->{b:.1f}")
    print("   hazard-multiplier jumps at defaults:", "; ".join(jumps) or "(none)")

out = os.path.join(os.path.dirname(__file__), "..", "docs", "demos", "gfc", "hazard_data.js")
with open(out, "w") as f:
    f.write("const HZ = " + json.dumps(data, separators=(",", ":")) + ";\n")
print("wrote", os.path.relpath(out), os.path.getsize(out) // 1024, "KB")
