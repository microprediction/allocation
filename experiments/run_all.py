"""Run the studies, record what each actually cost, and keep the output.

    python run_all.py --list
    python run_all.py --quick
    python run_all.py --full
    python run_all.py --only dimension,decompose

Output goes to `output/<name>.txt` with a header giving the wall time, so the
costs in README.md stay honest rather than remembered. A study that fails is
reported and does not stop the rest.
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
STUDIES = os.path.join(HERE, "studies")
OUT = os.path.join(HERE, "output")

# name -> (argv after the script, tier, one-line claim)
CATALOGUE = {
    "implied":            ([], "quick", "every portfolio is min-variance on some PD matrix"),
    "decompose":          ([], "quick", "HRP's gap split into information and rule"),
    "nonelliptical":      ([], "quick", "a market that separates tail risk from variance"),
    "psd_gamma":          ([], "quick", "where the implied covariance enters the PSD cone"),
    "bridge_distortion":  ([], "quick", "distortion falls monotonically along the dial"),
    "dimension":          ([], "slow",  "no class below n-1 parameters reproduces a portfolio"),
    "what_shrinkage":     ([], "slow",  "fits of four shrinkage families to HRP"),
    "closest":            ([], "slow",  "the closest sensible shrinkage is near-diagonal"),
    "confound":           ([], "slow",  "HRP against raw and filtered covariance"),
    "rank_deficient":     ([], "slow",  "the corner where the covariance is singular"),
    "robust":             ([], "slow",  "500 randomized structures"),
    "shrinkage_beats_hrp":([], "slow",  "Ledoit-Wolf long-only against HRP"),
    "nco_taper":          ([], "slow",  "NCO tracks the shrinkage frontier, HRP does not"),
    "taper_beats_hrp":    ([], "slow",  "HRP sits at the dominated corner"),
    "nested_best_case":   ([], "slow",  "HRP on a clean hierarchy, its best case"),
    "seriation_value":    ([], "slow",  "the seriation works, the split rule cannot use it"),
    "huge_universe":      (["5000", "104", "30"], "verylong", "5000 assets, 2y weekly: HRP wins"),
    "mhp":                (["40", "0.3"], "slow", "the multi-hypothesis method, benchmarked"),
    "thurstone_paired":   ([], "verylong", "the tilt is neutral on a stationary market"),
    "tail_test":          ([], "verylong", "the tilt on a regime market, both objectives"),
    "tilt_confirm":       ([], "verylong", "the tilt effect, first seed"),
    "tilt_replicate":     ([], "verylong", "the same, fresh seed"),
    "tilt_highdata":      ([], "verylong", "the high-coverage cells, third seed"),
}
TIERS = {"quick": ("quick",), "full": ("quick", "slow", "verylong"),
         "slow": ("quick", "slow")}


def run(name, argv):
    script = os.path.join(STUDIES, name + ".py")
    if not os.path.exists(script):
        return None, f"missing {script}"
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    p = subprocess.run([sys.executable, script] + argv, cwd=STUDIES,
                       capture_output=True, text=True)
    dt = time.time() - t0
    with open(os.path.join(OUT, name + ".txt"), "w") as fh:
        fh.write(f"# {name}: {CATALOGUE[name][2]}\n")
        fh.write(f"# wall time {dt:.1f}s, exit {p.returncode}\n\n")
        fh.write(p.stdout)
        if p.returncode != 0:
            fh.write("\n--- stderr ---\n" + p.stderr[-4000:])
    return dt, None if p.returncode == 0 else f"exit {p.returncode}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()

    if a.list:
        print(f"{'study':22s}{'tier':10s}what it establishes")
        for k, (_, tier, claim) in CATALOGUE.items():
            print(f"{k:22s}{tier:10s}{claim}")
        raise SystemExit

    if a.only:
        want = [s.strip() for s in a.only.split(",")]
    else:
        tiers = TIERS["full" if a.full else "quick"]
        want = [k for k, v in CATALOGUE.items() if v[1] in tiers]

    print(f"running {len(want)} studies, output to {OUT}\n")
    rows, failed = [], []
    for k in want:
        argv = CATALOGUE[k][0]
        print(f"  {k} ...", end="", flush=True)
        dt, err = run(k, argv)
        if err:
            print(f" FAILED ({err})"); failed.append(k)
        else:
            print(f" {dt:6.1f}s"); rows.append((k, dt))
    print(f"\n{'study':22s}{'seconds':>10s}")
    for k, dt in sorted(rows, key=lambda r: -r[1]):
        print(f"{k:22s}{dt:10.1f}")
    print(f"\ntotal {sum(d for _, d in rows):.0f}s over {len(rows)} studies")
    if failed:
        print(f"failed: {', '.join(failed)}")
        raise SystemExit(1)
