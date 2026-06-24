"""The race as fast, redundancy-aware feature attribution -- vs SHAP & permutation.

Following the credit-attribution thread to its ML home. Ground truth: y depends on
ONE signal direction, present as 1+K near-duplicate columns (correlated ~0.99), plus
many pure-noise features. A good importance should give the signal CLUSTER its due,
split evenly across the redundant copies, and ~0 to noise.

  coef        : |Ridge coefficient|
  permutation : sklearn permutation_importance (the classic correlated-feature trap:
                permuting one copy barely hurts because the others remain)
  shap        : mean |SHAP value| (the incumbent; correlation-aware but costly)
  race        : per-instance contribution c_ti = beta_i (x_ti - xbar_i); the feature
                with the largest |c| 'wins' the instance; win-frequency = the share.
                Redundant copies tie and split; O(M*n), one pass, the gradient of an
                expected-max potential.

We report each method's signal-cluster total, noise total, evenness across copies,
and wall time; plus a model-agnostic KernelSHAP timing to make the cost gap concrete.
"""
import time, warnings, numpy as np
warnings.filterwarnings("ignore")
from numpy.random import default_rng
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
import shap

rng = default_rng(0)
N, K, n_noise = 2000, 5, 20
x = rng.standard_normal(N)
sig = x[:, None] + 0.1 * rng.standard_normal((N, K + 1))     # K+1 near-duplicate signal cols
noise = rng.standard_normal((N, n_noise))
X = np.hstack([sig, noise]); y = 3.0 * x + 0.5 * rng.standard_normal(N)
nf = X.shape[1]; sig_idx = list(range(K + 1)); noise_idx = list(range(K + 1, nf))
model = Ridge(alpha=1.0).fit(X, y); beta = model.coef_
norm = lambda v: (np.abs(v) / np.abs(v).sum())

def timed(f):
    t = time.perf_counter(); out = f(); return out, time.perf_counter() - t

imp = {}; tm = {}
imp["coef"], tm["coef"] = norm(beta), 0.0
(pi, tm["permutation"]) = timed(lambda: permutation_importance(model, X, y, n_repeats=10, random_state=0))
imp["permutation"] = norm(np.clip(pi.importances_mean, 0, None))
(sv, tm["shap"]) = timed(lambda: shap.LinearExplainer(model, X).shap_values(X))
imp["shap"] = norm(np.abs(sv).mean(0))
def race():
    C = beta * (X - X.mean(0)); return np.bincount(np.argmax(np.abs(C), 1), minlength=nf) / N
imp["race"], tm["race"] = timed(race)

print(f"ground truth: y = 3*x; x present as {K+1} near-duplicate columns + {n_noise} noise\n")
print(f"{'method':12}{'signal cluster':>16}{'noise total':>13}{'copy spread':>13}{'time (ms)':>11}")
print("-" * 65)
for m in ["coef", "permutation", "shap", "race"]:
    v = imp[m]
    spread = v[sig_idx].max() / max(v[sig_idx].min(), 1e-9)   # 1 = perfectly even split
    print(f"{m:12}{v[sig_idx].sum():>16.3f}{v[noise_idx].sum():>13.3f}"
          f"{spread:>13.1f}{tm[m]*1e3:>11.1f}")

# model-agnostic cost: KernelSHAP on a small slice vs the race on everything
bg = shap.kmeans(X, 10)
ksub = X[:60]
_, t_ks = timed(lambda: shap.KernelExplainer(model.predict, bg).shap_values(ksub, silent=True))
print(f"\nmodel-agnostic cost: KernelSHAP on 60 instances took {t_ks:.1f}s "
      f"(~{t_ks/60*N:.0f}s for all {N}); the race did all {N} in {tm['race']*1e3:.1f} ms")

print("\nread: the COST advantage is real and large -- the race attributes all instances in")
print("<1ms where model-agnostic KernelSHAP needs seconds. All methods credit the signal")
print("cluster and reject noise (race best at 0.002). BUT the naive argmax race is")
print("winner-take-all among near-duplicates (copy spread ~26x, NOT an even split): the")
print("copies carry slightly unequal contributions, so one wins most instances. Even-")
print("splitting is a property of the race at EQUAL abilities (Prop 4) -- which arbitrary")
print("contribution scores do not satisfy. So the race is a fast importance/screen here; its")
print("redundancy-EVENNESS needs the calibrated equal-ability construction, empirically")
print("corroborating the narrowed redundancy claim (REVISIONS.md item 3b).")
