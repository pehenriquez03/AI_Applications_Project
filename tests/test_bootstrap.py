import hashlib

import numpy as np
import pandas as pd

from research_engine.economics.bootstrap import (bootstrap_index_matrix, joint_paths,
                                                  stationary_bootstrap_indices)


def _mean_run_length(idx_matrix, n_obs):
    runs = 0
    total = 0
    for row in idx_matrix:
        cont = (row[1:] == (row[:-1] + 1) % n_obs)
        runs += int((~cont).sum()) + 1
        total += len(row)
    return total / runs


def test_shape():
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(300, 3)), columns=list("abc"))
    out = joint_paths(df, n_paths=7, horizon_days=50, mean_block_days=20, seed=1)
    assert out.shape == (7, 50, 3)


def test_single_path_indices_in_range_and_wrap():
    rng = np.random.default_rng(3)
    idx = stationary_bootstrap_indices(10, 500, 50, rng)
    assert idx.min() >= 0 and idx.max() < 10 and len(idx) == 500
    # long blocks on a tiny sample must wrap around
    assert np.any(idx[1:] < idx[:-1])


def test_mean_block_one_is_iid():
    idx = bootstrap_index_matrix(5000, 40, 1000, 1, seed=11)
    assert _mean_run_length(idx, 5000) < 1.01


def test_mean_block_126_run_length():
    idx = bootstrap_index_matrix(3000, 300, 6000, 126, seed=5)
    m = _mean_run_length(idx, 3000)
    assert 126 * 0.85 <= m <= 126 * 1.15


def test_same_seed_identical_and_different_seed_differs():
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(400, 2)))
    a = joint_paths(df, 20, 100, 30, seed=9)
    b = joint_paths(df, 20, 100, 30, seed=9)
    c = joint_paths(df, 20, 100, 30, seed=10)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_path_i_independent_of_n_paths():
    a = bootstrap_index_matrix(400, 10, 100, 30, seed=2)
    b = bootstrap_index_matrix(400, 25, 100, 30, seed=2)
    assert np.array_equal(a, b[:10])


def test_columns_share_dates():
    x = np.random.default_rng(1).normal(size=500)
    df = pd.DataFrame({"x": x, "y": 2 * x})
    out = joint_paths(df, 30, 200, 40, seed=4)
    assert np.array_equal(out[:, :, 1], 2 * out[:, :, 0])


def test_index_hash_stable():
    h = lambda m: hashlib.sha256(m.tobytes()).hexdigest()
    assert h(bootstrap_index_matrix(300, 5, 60, 20, 8)) == h(bootstrap_index_matrix(300, 5, 60, 20, 8))
