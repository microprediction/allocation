"""Reject misspelled bridge algorithms without consuming observations (#144)."""
import pickle

import numpy as np
import pytest

from allocation import SchurBridge, StreamingSchurBridge
from allocation._schur.bridge import bridge_weights

COV = np.array([[2., .8, .4, .1], [.8, 1.5, .2, .3],
                [.4, .2, 1.2, .5], [.1, .3, .5, 1.8]])
PART = [np.array([0, 1]), np.array([2, 3])]
INVALID = ['typo', '', 'STACK', None, True, 1, ['stack'], np.array(['stack'])]


@pytest.mark.parametrize('conditioning', ['siblings', 'all', 'factor'])
@pytest.mark.parametrize('outer', INVALID)
def test_low_level_rejects_invalid_outer(conditioning, outer):
    with pytest.raises(ValueError, match="outer must be 'stack' or 'optimize'"):
        bridge_weights(COV, PART, gamma=.4, eta=.6,
                       conditioning=conditioning, outer=outer)


@pytest.mark.parametrize('method', ['fit', 'partial_fit'])
@pytest.mark.parametrize('fitted', [False, True])
def test_batch_rejects_before_state_mutation(method, fitted):
    X = np.random.default_rng(144).normal(size=(30, 4))
    est = SchurBridge(n_clusters=2, conditioning='all')
    if fitted:
        est.fit(X)
    est.set_params(outer='typo')
    before = pickle.dumps(est.__dict__)
    with pytest.raises(ValueError, match='outer must'):
        getattr(est, method)(X[:1])
    assert pickle.dumps(est.__dict__) == before


@pytest.mark.parametrize('warm', [False, True])
def test_streaming_rejects_before_observation_or_partition_changes(warm):
    est = StreamingSchurBridge(n_clusters=2, conditioning='all', min_obs=2)
    if warm:
        for row in np.random.default_rng(144).normal(size=(30, 4)):
            est.learn_one(dict(zip('ABCD', row)))
    est.outer = 'typo'
    before = pickle.dumps(est.__dict__)
    with pytest.raises(ValueError, match='outer must'):
        est.learn_one(dict(zip('ABCD', [.2, .3, -.1, .4])))
    assert pickle.dumps(est.__dict__) == before


@pytest.mark.parametrize('conditioning', ['siblings', 'all', 'factor'])
@pytest.mark.parametrize('outer', ['stack', 'optimize'])
def test_valid_modes_and_numpy_strings_preserved(conditioning, outer):
    expected = bridge_weights(COV, PART, gamma=.4, eta=.6,
                              conditioning=conditioning, outer=outer)
    actual = bridge_weights(COV, PART, gamma=.4, eta=.6,
                            conditioning=conditioning, outer=np.str_(outer))
    np.testing.assert_array_equal(actual, expected)
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual.sum(), 1.)
