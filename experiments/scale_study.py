"""Scalability: the regime where Thurstone earns its keep.

With more names than observations (n > T) the sample covariance is rank-deficient,
so dense inversion allocators (minimum variance, max diversification) are
*undefined*. The factor-form Thurstone race never inverts and costs O(M n k); the
Woodbury factor optimizers stay well-posed via an idiosyncratic floor. This script
reports, as n grows past a fixed window T, which methods remain well-posed and how
long a single fit takes.
"""

import time
import numpy as np

from allocation import (ThurstonePortfolio, FactorMinimumVariance, InverseVariance,
                        HierarchicalRiskParity, MinimumVariance)
from allocation.convex import is_singular


def factor_returns(rng, n, T, k=5):
    B = rng.standard_normal((n, k)) * 0.3
    F = rng.standard_normal((T, k))
    return F @ B.T + 0.5 * rng.standard_normal((T, n))   # (T, n)


def timed(fac, R):
    t0 = time.perf_counter()
    fac().fit(R)
    return time.perf_counter() - t0


def main():
    rng = np.random.default_rng(0)
    T, k = 250, 5
    print(f"window T={T} observations, {k}-factor data; fit time (s) per method\n")
    hdr = (f"{'n':>6}{'n>T?':>6}{'MinVar':>9}{'Thurstone(f={})'.format(k):>16}"
           f"{'FactorMV':>10}{'InvVar':>8}{'HRP':>8}")
    print(hdr); print("-" * len(hdr))
    for n in (50, 100, 250, 500, 1000, 2000, 3000):
        R = factor_returns(rng, n, T, k)
        cov = np.cov(R, rowvar=False)
        sing = is_singular(cov)
        mv = "undef." if sing else f"{timed(lambda: MinimumVariance(shrinkage=0.0), R):.3f}"
        th = timed(lambda: ThurstonePortfolio(calib="diagonal", target="equal", factors=k), R)
        fm = timed(lambda: FactorMinimumVariance(factors=k), R)
        iv = timed(lambda: InverseVariance(), R)
        try:
            hrp = f"{timed(lambda: HierarchicalRiskParity(), R):.3f}"
        except Exception:
            hrp = "err"
        print(f"{n:>6}{('yes' if n > T else 'no'):>6}{mv:>9}{th:>16.3f}{fm:>10.3f}{iv:>8.3f}{hrp:>8}")

    print("\nMinVar/MaxDiv are undefined once n>T (singular covariance); the factor-form")
    print("Thurstone race and the Woodbury factor optimizers remain well-posed and scale.")


if __name__ == "__main__":
    main()
