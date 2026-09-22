"""Restriction done cleanly: keep the seriation, drop a whole block.

Recomputing HRP on an arbitrary subset rebuilds the tree, so the target is not
the same rule on fewer assets and no predictor can be judged fairly. Dropping
a complete subtree fixes that: every surviving split is untouched, so the
restricted portfolio is unambiguous.

It also makes a prediction. HRP's weight is a product of split fractions along
a path. Deleting a sibling subtree leaves every split inside the survivor
unchanged, so all surviving weights scale by one common factor and rescaling
is EXACT. That is a property of the rule, not a measurement, and the first
table checks it.

Minimum variance has no such property: its restriction depends on the
sub-covariance. That is where a choice model has something to do.
"""
import sys
import numpy as np
import winning
from confound import tree as corr_tree, quasi_diag, min_var
from robust import random_structure, mv_lo


def build_tree(order):
    """Nested tuples by repeated halving: the tree HRP actually uses."""
    order = list(order)
    if len(order) <= 1:
        return order
    h = len(order) // 2
    return (build_tree(order[:h]), build_tree(order[h:]))


def prune(t, drop):
    """Remove a set of assets, keeping every surviving split where it was.
    A node whose sibling vanishes collapses into its survivor."""
    if isinstance(t, list):
        keep = [i for i in t if i not in drop]
        return keep if keep else None
    a, b = prune(t[0], drop), prune(t[1], drop)
    if a is None:
        return b
    if b is None:
        return a
    return (a, b)


def leaves(t):
    return list(t) if isinstance(t, list) else leaves(t[0]) + leaves(t[1])


def hrp_on_pruned(S, t, index):
    """HRP down a given tree, with `index` mapping asset id to a row of S."""
    w = {i: 1.0 for i in leaves(t)}

    def rec(node):
        if isinstance(node, list):
            return
        l, r = leaves(node[0]), leaves(node[1])
        vl = _cvar(S, [index[i] for i in l])
        vr = _cvar(S, [index[i] for i in r])
        a = 1.0 - vl / (vl + vr)
        for i in l:
            w[i] *= a
        for i in r:
            w[i] *= 1.0 - a
        rec(node[0]); rec(node[1])

    rec(t)
    ids = sorted(w)
    v = np.array([w[i] for i in ids])
    return ids, v / v.sum()


def hrp_on_tree(S, order):
    """Recursive bisection down a FIXED order, returning weights and the split
    tree so a subtree can be identified."""
    n = len(order)
    w = np.ones(S.shape[0])
    nodes = []
    stack = [list(order)]
    while stack:
        nxt = []
        for c in stack:
            if len(c) <= 1:
                continue
            h = len(c) // 2
            l, r = c[:h], c[h:]
            nodes.append((l, r))
            vl, vr = _cvar(S, l), _cvar(S, r)
            a = 1.0 - vl / (vl + vr)
            w[l] *= a
            w[r] *= 1.0 - a
            nxt += [l, r]
        stack = nxt
    return w / w.sum(), nodes


def _cvar(S, idx):
    sub = S[np.ix_(idx, idx)]
    iv = 1.0 / np.diag(sub)
    iv /= iv.sum()
    return float(iv @ sub @ iv)


def factor_form(C, k=3):
    d = np.sqrt(np.clip(np.diag(C), 1e-300, None))
    R = C / np.outer(d, d)
    ev, U = np.linalg.eigh((R + R.T) / 2)
    i = np.argsort(ev)[::-1][:k]
    V = U[:, i] * np.sqrt(np.clip(ev[i], 0.0, None))
    D = np.clip(1.0 - (V ** 2).sum(1), 1e-6, None)
    sc = np.sqrt((V ** 2).sum(1) + D)
    return V / sc[:, None], D / sc ** 2


def run(n=256, T=1000, draws=30, seed=9, depth=2):
    rng = np.random.default_rng(seed)
    out = {a: {"race": [], "race+corr": [], "rescale": []} for a in ("HRP", "min-var")}
    for _ in range(draws):
        Sig, _ = random_structure(rng, n)
        X = rng.normal(size=(T, n)) @ np.linalg.cholesky(Sig).T
        S = np.cov(X, rowvar=False)
        Z, _ = corr_tree(S)
        order = quasi_diag(Z, n)

        # drop one complete subtree at the chosen depth
        block = list(order)
        for _ in range(depth):
            h = len(block) // 2
            block = block[:h] if rng.random() < 0.5 else block[h:]
        keep = np.array(sorted(set(order) - set(block)))
        sub_order = [i for i in order if i in set(keep)]

        full_tree = build_tree(order)
        idx_full = {i: i for i in order}
        ids_f, wf_vec = hrp_on_pruned(S, full_tree, idx_full)
        w_full = np.zeros(n); w_full[ids_f] = wf_vec

        pruned = prune(full_tree, set(block))
        pos = {int(a): j for j, a in enumerate(keep)}
        ids_s, ws_vec = hrp_on_pruned(S[np.ix_(keep, keep)], pruned, pos)
        w_sub = np.zeros(len(keep))
        for i, val in zip(ids_s, ws_vec):
            w_sub[pos[i]] = val
        mv_full, mv_sub = mv_lo(S), mv_lo(S[np.ix_(keep, keep)])

        V, D = factor_form(S, 3)
        for name, wf, ws in (("HRP", w_full, w_sub), ("min-var", mv_full, mv_sub)):
            resc = wf[keep] / wf[keep].sum()
            out[name]["rescale"].append(float(np.abs(resc - ws).sum()))

            a = np.asarray(winning.calibrate_abilities(np.maximum(wf, 1e-12)), float)
            p = winning.race_probabilities(a[keep])
            p = np.asarray(p[0] if isinstance(p, tuple) else p, float)
            out[name]["race"].append(float(np.abs(p / p.sum() - ws).sum()))

            # calibrated and restricted under the same estimated correlation
            ac = np.asarray(winning.calibrate_abilities(
                np.maximum(wf, 1e-12), V=V, D=D), float)
            pc = winning.race_probabilities(ac[keep], V=V[keep], D=D[keep])
            pc = np.asarray(pc[0] if isinstance(pc, tuple) else pc, float)
            out[name]["race+corr"].append(float(np.abs(pc / pc.sum() - ws).sum()))
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    print(f"universe {n}, one complete subtree dropped, seriation fixed, 30 draws.")
    print("L1 error in predicting the restricted portfolio\n")
    out = run(n=n)
    def wil(p, nn, z=1.96):
        dd = 1 + z*z/nn; c = (p + z*z/(2*nn))/dd
        h = z*((p*(1-p)/nn + z*z/(4*nn*nn))**0.5)/dd
        return max(c-h,0), min(c+h,1)
    print(f"{'allocator':14s}{'rescaling':>11s}{'race':>9s}{'race+corr':>12s}"
          f"{'corr beats rescale':>22s}")
    for name, d in out.items():
        q = np.asarray(d["rescale"]); r = np.asarray(d["race"]); c = np.asarray(d["race+corr"])
        w = float(np.mean(c < q)); lo, hi = wil(w, len(c))
        print(f"{name:14s}{np.median(q):11.4f}{np.median(r):9.4f}{np.median(c):12.4f}"
              f"{f'{w:.0%} [{lo:.0%},{hi:.0%}]':>22s}")
