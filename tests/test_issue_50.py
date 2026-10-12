"""#50: KeyedEwmaCovariance.matrix must stay positive semidefinite when assets
leave and rejoin the universe."""
import numpy as np

from allocation.keyed import KeyedEwmaCovariance, StreamingMinimumVariance


def _min_eig(C):
    return float(np.linalg.eigvalsh(C)[0])


def test_absent_asset_does_not_leave_an_indefinite_matrix():
    c = KeyedEwmaCovariance(halflife=1)
    c.learn_one({"a": 0.0, "b": 0.0})
    c.learn_one({"a": 2.0, "b": 2.0})
    for _ in range(10):
        c.learn_one({"a": 1.0})
    C = c.matrix(["a", "b"])
    assert _min_eig(C) > 0
    w = np.array([2.0, -1.0])
    assert w @ C @ w > 0
    # variances are not altered by the repair
    assert np.allclose(np.diag(C), [c.cov[("a", "a")], c.cov[("b", "b")]])
    # rejoining for one observation still yields a covariance
    c.learn_one({"a": 1.0, "b": 0.5})
    assert _min_eig(c.matrix(["a", "b"])) > 0


def test_random_leave_rejoin_stays_psd():
    rng = np.random.default_rng(0)
    ids = list("abcdef")
    c = KeyedEwmaCovariance(halflife=3)
    for _ in range(300):
        present = [k for k in ids if rng.random() < 0.6] or ["a"]
        c.learn_one({k: float(rng.standard_normal()) for k in present})
        C = c.matrix(ids)
        assert _min_eig(C) >= -1e-12 * np.max(np.diag(C))


def test_synchronous_universe_matches_batch_ewma():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 4)) @ rng.standard_normal((4, 4))
    ids = ["w", "x", "y", "z"]
    c = KeyedEwmaCovariance(halflife=10)
    a = 1.0 - 0.5 ** (1.0 / 10)
    mean, cov = X[0].copy(), np.zeros((4, 4))
    c.learn_one(dict(zip(ids, X[0])))
    for row in X[1:]:
        c.learn_one(dict(zip(ids, row)))
        mean = (1 - a) * mean + a * row
        dev = row - mean
        cov = (1 - a) * cov + a * np.outer(dev, dev)
    assert np.allclose(c.matrix(ids), cov)


def test_streaming_min_variance_has_nonnegative_risk_after_absence():
    m = StreamingMinimumVariance(halflife=1, min_obs=1)
    m.learn_one({"a": 0.0, "b": 0.0})
    m.learn_one({"a": 2.0, "b": 2.0})
    for _ in range(10):
        m.learn_one({"a": 1.0})
    m.learn_one({"a": 1.0, "b": 1.0})
    ids = sorted(m.weights)
    w = np.array([m.weights[k] for k in ids])
    assert np.all(np.isfinite(w))
    assert w @ m._cov.matrix(ids) @ w >= 0
