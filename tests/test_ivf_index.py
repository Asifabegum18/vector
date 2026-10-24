import numpy as np
import pytest
from src.distance import squared_l2, nearest_indices
from src.ivf_index import IVFIndex

def test_squared_l2():
    np.testing.assert_allclose(squared_l2(np.array([0, 0]), np.array([[1, 0], [0, 2]])), [1, 4])

def test_nearest_indices():
    ix, dist = nearest_indices(np.array([0, 0]), np.array([[4, 0], [1, 0], [2, 0]]), 2)
    assert ix.tolist() == [1, 2]
    np.testing.assert_allclose(dist, [1, 4])

def test_every_vector_in_exactly_one_inverted_list():
    x = np.random.default_rng(3).normal(size=(200, 8)).astype(np.float32)
    index = IVFIndex(8, random_state=7, max_iter=20).fit(x)
    rows = np.concatenate(index.inverted_lists)
    assert sorted(rows.tolist()) == list(range(len(x)))

def test_search_sorted_and_full_probe_contains_exact_query():
    rng = np.random.default_rng(5)
    x = np.vstack([rng.normal(-4, .1, (80, 4)), rng.normal(4, .1, (80, 4))]).astype(np.float32)
    ids = np.arange(1000, 1000 + len(x))
    index = IVFIndex(4, random_state=2, max_iter=30).fit(x, ids)
    result = index.search(x[3], k=5, nprobe=4)
    assert len(result.product_ids) == 5
    assert result.candidates_examined == len(x)
    assert result.product_ids[0] == ids[3]
    assert np.all(result.distances[:-1] <= result.distances[1:])

def test_invalid_nprobe_and_dimension():
    x = np.random.default_rng(1).normal(size=(30, 4)).astype(np.float32)
    index = IVFIndex(3, max_iter=10).fit(x)
    with pytest.raises(ValueError):
        index.search(x[0], nprobe=0)
    with pytest.raises(ValueError):
        index.search(np.zeros(5, dtype=np.float32))
