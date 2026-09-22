"""Prove that a result does not depend on how the work was divided.

Run the same draws as one shard and as three, and compare every recorded
number. A rule that quietly depends on what was solved before it is a rule
whose merged table depends on how many cores the machine had.

Bit-identity is the standard for everything except the last ULP. Shards run in
separate processes, and BLAS reductions can differ in the final bit with array
alignment, which varies between processes and is not something this code
controls. So the bar is four units in the last place, and the report prints
the gap in ULPs rather than a raw tolerance, because a genuine state leak is
orders of magnitude larger than that and will not hide under it. Measured: the
defect below showed up at 3.5e-03, or about ten trillion ULPs.

This check has already earned its place. Caching the cvxpy Problem and
re-solving it through a Parameter carried solver state between draws, which
moved the parent portfolio by about 1e-8. Small, except the parent is the
input to every restriction rule, so `proportional` shifted by 1.3e-04 and
`race+factor` by 3.5e-03 while `equal weight` stayed bit-identical. That
signature, the data-free rows exact and the parent-dependent rows drifting,
is what to look for if this ever fails again.

    python check_sharding.py
"""
import glob
import json
import math
import os
import subprocess
import sys
import tempfile

ARGS = ["--scale", "mid", "--n", "80", "--m", "20", "--draws", "6",
        "--k", "2", "--Ts", "30"]


def run_config(out, shards):
    for i in range(shards):
        subprocess.run(
            [sys.executable, "run.py", *ARGS, "--shard", str(i),
             "--shards", str(shards), "--tag", f"inv{shards}", "--out", out],
            check=True, capture_output=True,
            env={**os.environ, "OMP_NUM_THREADS": "1",
                 "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    rows = {}
    for f in glob.glob(f"{out}/inv{shards}-shard*.json"):
        for r in json.loads(open(f).read())["rows"]:
            rows[r["draw"]] = r
    return rows


def main():
    with tempfile.TemporaryDirectory() as out:
        print("running 6 draws as 1 shard, then as 3")
        a, b = run_config(out, 1), run_config(out, 3)

    keys = [k for k in a[0] if isinstance(a[0][k], float)]
    bad = []
    for k in keys:
        d = max(abs(a[i][k] - b[i][k]) for i in a)
        scale = max(abs(a[i][k]) for i in a) or 1.0
        ulps = d / (math.ulp(scale) or 1.0)
        if ulps > 4:
            bad.append((k, d, ulps))
        print(f"  {'FAIL' if ulps > 4 else 'ok  '}  {k:26s} "
              f"max difference {d:.2e}  ({ulps:.0f} ulp)")

    if bad:
        print(f"\n{len(bad)} quantities depend on the sharding by more than "
              "rounding. The merged table is not reproducible.")
        for k, d, u in bad:
            print(f"    {k}: {d:.2e} ({u:.0f} ulp)")
        return 1
    print(f"\nall {len(keys)} quantities agree to within rounding across "
          f"{len(a)} draws. Shards can be spread over any number of workers "
          "or hosts and merged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
