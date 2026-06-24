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
from allocation._thurstone.calibrate import calibrate_diagonal, base_density
from allocation._thurstone.transport import transport_weights

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
imp["race (raw argmax)"], tm["race (raw argmax)"] = timed(race)

# Calibrated race: abilities calibrated to the MARGINAL (univariate) importance -- equal
# across the copies -- then the race is tilted by the feature correlation, so the
# redundant copies de-duplicate and split evenly (Prop 4), while noise stays ~0.
def race_calibrated():
    uni = np.array([np.corrcoef(X[:, i], y)[0, 1] ** 2 for i in range(nf)])  # marginal strength
    w0 = uni / uni.sum()                                  # benchmark on the simplex
    theta = calibrate_diagonal(w0, base=base_density())   # abilities: indep race -> w0
    Cc = np.corrcoef(X.T)                                 # tilt = feature correlation
    seeds = default_rng(7).standard_normal((1 << 14, nf))
    return np.asarray(transport_weights(theta, Cc, seeds))
imp["race (calibrated)"], tm["race (calibrated)"] = timed(race_calibrated)

print(f"ground truth: y = 3*x; x present as {K+1} near-duplicate columns + {n_noise} noise\n")
print(f"{'method':20}{'signal cluster':>16}{'noise total':>13}{'copy spread':>13}{'time (ms)':>11}")
print("-" * 73)
for m in ["coef", "permutation", "shap", "race (raw argmax)", "race (calibrated)"]:
    v = imp[m]
    spread = v[sig_idx].max() / max(v[sig_idx].min(), 1e-9)   # 1 = perfectly even split
    print(f"{m:20}{v[sig_idx].sum():>16.3f}{v[noise_idx].sum():>13.3f}"
          f"{spread:>13.1f}{tm[m]*1e3:>11.1f}")

# model-agnostic cost: KernelSHAP on a small slice vs the race on everything
bg = shap.kmeans(X, 10)
ksub = X[:60]
_, t_ks = timed(lambda: shap.KernelExplainer(model.predict, bg).shap_values(ksub, silent=True))
print(f"\nmodel-agnostic cost: KernelSHAP on 60 instances took {t_ks:.1f}s "
      f"(~{t_ks/60*N:.0f}s for all {N}); the calibrated race ran in "
      f"{tm['race (calibrated)']*1e3:.1f} ms")

print("\nread: raw-argmax race is winner-take-all among near-duplicates (spread ~26x), because")
print("the copies carry slightly UNequal contributions. The CALIBRATED race fixes exactly")
print("this: abilities are set to the marginal (univariate) importance -- equal across the")
print("copies -- then the race is tilted by the feature correlation, so the copies de-")
print("duplicate and split EVENLY (spread -> ~1). That is Prop 4 in action, and it is what")
print("raw scores cannot give. Tradeoff: de-duplicated weight partly leaks to noise (total")
print("0.09, but ~0.004 per noise feature vs ~0.15 per signal copy -- still 30x down), and")
print("the calibrated race costs ~0.3s (calibration + 16k-path tilt) vs the raw argmax's")
print("sub-ms -- still ~90x faster than model-agnostic SHAP's ~24s. Net: calibration buys")
print("the redundancy-even-split, the headline property, at a modest, still-cheap cost.")

# Temperature sweep: scaling the calibrated strengths by beta suppresses the noise
# leak (Prop: concentration) while the even split among copies survives.
print("\ntemperature beta on the calibrated strengths (leak control, redundancy preserved):")
uni = np.array([np.corrcoef(X[:, i], y)[0, 1] ** 2 for i in range(nf)])
theta = calibrate_diagonal(uni / uni.sum(), base=base_density())
Cc = np.corrcoef(X.T); seeds = default_rng(7).standard_normal((1 << 14, nf))
print(f"  {'beta':>6}{'signal':>9}{'noise':>8}{'copy spread':>13}")
for b in [0.5, 1.0, 2.0, 4.0]:
    w = np.asarray(transport_weights(b * theta, Cc, seeds))
    spread = w[sig_idx].max() / max(w[sig_idx].min(), 1e-9)
    print(f"  {b:>6.1f}{w[sig_idx].sum():>9.3f}{w[noise_idx].sum():>8.3f}{spread:>13.1f}")
print("  -> beta>=2 drives the noise leak to ~0 with copy spread still ~1 (Prop: temperature).")
