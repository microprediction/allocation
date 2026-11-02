"""A Monte Carlo path request is an integer count, never a truncation."""
import numpy as np
import pytest

from allocation import StreamingThurstone, ThurstonePortfolio, tilt_weights
from allocation._thurstone.transport import path_budget


@pytest.mark.parametrize('value', [True, False, np.bool_(True), '8', None, 2.1, 4.1, 7.9,
                                 np.float64(8.9), float('nan'), float('inf'),
                                 -float('inf'), 8+0j, [8], np.array(8), 0, 1, -2])
def test_invalid_counts(value):
    with pytest.raises(ValueError, match='n_paths'):
        path_budget(value)


@pytest.mark.parametrize('value, expected', [(2, 2), (3, 4), (4, 4), (5, 8),
    (65535, 65536), (65536, 65536), (65537, 131072), (np.int64(5), 8),
    (np.uint64(8), 8), (8.0, 8), (np.float64(5), 8),
    (2**53+1, 2**54), (2**100+1, 2**101)])
def test_valid_counts_round_up_exactly(value, expected):
    assert path_budget(value) == expected


@pytest.mark.parametrize('value', [4.1, '8', True, np.float64(8.9), float('nan'), float('inf')])
@pytest.mark.parametrize('consumer', ['function', 'batch', 'stream'])
def test_public_consumers_reject(value, consumer):
    with pytest.raises(ValueError, match='n_paths'):
        if consumer == 'function':
            tilt_weights([.5, .5], np.eye(2), n_paths=value)
        elif consumer == 'batch':
            ThurstonePortfolio(target='equal', n_paths=value).fit(np.eye(2))
        else:
            StreamingThurstone(n_paths=value)


def test_valid_five_paths_uses_eight_in_consumers():
    batch = ThurstonePortfolio(target='equal', n_paths=5).fit(np.eye(2))
    assert batch._seeds.shape == (8, 2)
    assert StreamingThurstone(n_paths=5).n_paths == 8
    np.testing.assert_array_equal(
        tilt_weights([.5, .5], np.eye(2), n_paths=5),
        tilt_weights([.5, .5], np.eye(2), n_paths=8))
