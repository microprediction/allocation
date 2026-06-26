"""Outside option: a null competitor that absorbs the de-duplication leak.

The one residual cost of the calibrated race is that weight freed when a strong
cluster de-duplicates leaks to weak contributors (simplex efficiency must put the
mass somewhere). Temperature (Prop: temperature) fixes this by sharpening toward a
hard argmax. An alternative that keeps the race smooth is an OUTSIDE OPTION: add a
null competitor with strength c (ability a0, independent noise). The named shares
then sum to 1 - w0, and w0 = P(null wins) absorbs the mass no real contributor
deserves. If a0 sits between the signal and noise abilities, the null beats the
noise (absorbing the leak into w0) but loses to the signal cluster (which keeps its
even split). Convention: package min-wins, so SMALLER ability is stronger.

Setup mirrors feature_attribution.py: one signal direction as K+1 near-duplicate
columns + noise. We sweep the outside-option ability a0 and report the named signal
cluster, named noise, the absorbed mass w0, and the copy spread (1 = even split).
"""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from numpy.random import default_rng
from allocation._thurstone.calibrate import calibrate_diagonal, base_density
from allocation._thurstone.transport import transport_weights

rng = default_rng(0)
N, K, n_noise = 2000, 5, 20
x = rng.standard_normal(N)
sig = x[:, None] + 0.12 * rng.standard_normal((N, K + 1))
X = np.hstack([sig, rng.standard_normal((N, n_noise))]); y = 3.0 * x + 0.5 * rng.standard_normal(N)
nf = X.shape[1]; sig_idx = list(range(K + 1)); noise_idx = list(range(K + 1, nf))

uni = np.array([np.corrcoef(X[:, i], y)[0, 1] ** 2 for i in range(nf)])
theta = calibrate_diagonal(uni / uni.sum(), base=base_density())   # smaller ability = stronger
Cc = np.corrcoef(X.T)
M = 1 << 14
a_sig, a_noise = theta[sig_idx].mean(), theta[noise_idx].mean()    # strong (small) vs weak (large)
print(f"calibrated abilities: signal mean {a_sig:.2f} (strong), noise mean {a_noise:.2f} (weak)\n")

def race(a0=None):
    if a0 is None:
        w = np.asarray(transport_weights(theta, Cc, default_rng(7).standard_normal((M, nf))))
        return w, 0.0
    th = np.append(theta, a0)
    C = np.eye(nf + 1); C[:nf, :nf] = Cc                           # null independent of the field
    w = np.asarray(transport_weights(th, C, default_rng(7).standard_normal((M, nf + 1))))
    return w[:nf], w[nf]

print(f"{'outside a0':>12}{'signal':>9}{'noise':>8}{'w0 (null)':>11}{'copy spread':>13}")
print("-" * 53)
w, _ = race(None)
spread = w[sig_idx].max() / max(w[sig_idx].min(), 1e-9)
print(f"{'none':>12}{w[sig_idx].sum():>9.3f}{w[noise_idx].sum():>8.3f}{0.0:>11.3f}{spread:>13.1f}")
# sweep the null ability from just below the noise level down toward the signal level
for frac in [0.0, 0.25, 0.5, 0.75]:
    a0 = a_noise - frac * (a_noise - a_sig)
    w, w0 = race(a0)
    spread = w[sig_idx].max() / max(w[sig_idx].min(), 1e-9)
    print(f"{a0:>12.2f}{w[sig_idx].sum():>9.3f}{w[noise_idx].sum():>8.3f}{w0:>11.3f}{spread:>13.1f}")

print("\nread (honest, nuanced): the null preserves the even split (spread ~1) at every")
print("strength and adds an interpretable baseline share w0 (unexplained / cash / market-")
print("mode for portfolios, baseline for attribution). It absorbs mass with only a MILD bias")
print("toward the weak -- at the strongest null, w0=0.22 is drawn ~0.18 from the signal and")
print("~0.04 from the noise (noise gives ~2x its field share, so the leak softens, but the")
print("dominant signal is shaved too). So the outside option is a smooth, interpretable floor,")
print("complementary to temperature -- NOT a clean targeted leak remover. For pure leak")
print("suppression, temperature (Prop: temperature) is the sharper tool; the null's value is")
print("the meaningful w0 mass and a smooth alternative that never hard-argmaxes.")
