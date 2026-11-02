"""Invalid prior controls must fail before consuming observations (#143)."""
import pickle

import numpy as np
import pytest

from allocation import HierarchicalRiskParity, SchurBridge, SchurComplementary
from allocation._schur.seriation import affinity, seriate

BAD = [True, np.bool_(False), np.nan, np.inf, -np.inf, -0.1, 1.1, '0.5', [0.5], np.array(0.5)]
X = np.random.default_rng(812).normal(size=(30, 4))
P = np.eye(4)

@pytest.mark.parametrize('bad', BAD)
@pytest.mark.parametrize('fn', [affinity, seriate])
@pytest.mark.parametrize('prior', [None, P])
def test_shared_boundary(bad, fn, prior):
    with pytest.raises(ValueError, match='prior_weight'):
        fn(np.cov(X.T), prior=prior, prior_weight=bad)

@pytest.mark.parametrize('bad', BAD)
@pytest.mark.parametrize('cls', [SchurComplementary, HierarchicalRiskParity, SchurBridge])
@pytest.mark.parametrize('method', ['fit', 'partial_fit'])
def test_reject_before_state_mutation(bad, cls, method):
    est = cls(prior=P, prior_weight=.5).fit(X)
    est.set_params(prior_weight=bad)
    before = pickle.dumps(est)
    with pytest.raises(ValueError, match='prior_weight'):
        getattr(est, method)(X[:3])
    assert pickle.dumps(est) == before

@pytest.mark.parametrize('bad', BAD)
def test_fixed_clusters_do_not_hide_invalid_configuration(bad):
    est = SchurBridge(clusters=[0, 0, 1, 1], prior_weight=bad)
    before = pickle.dumps(est)
    with pytest.raises(ValueError, match='prior_weight'):
        est.fit(X)
    assert pickle.dumps(est) == before

@pytest.mark.parametrize('weight', [0, 1, .25, np.float32(.5), np.float64(.8), np.int64(1)])
def test_valid_blend(weight):
    cov = np.cov(X.T)
    base = affinity(cov)
    expected = (1-weight)*base + weight*P
    np.fill_diagonal(expected, 0)
    np.testing.assert_array_equal(affinity(cov, prior=P, prior_weight=weight), expected)
