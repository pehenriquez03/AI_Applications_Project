"""Politis-Romano stationary bootstrap with a joint date sequence.

Every column of a DataFrame is resampled with the SAME index sequence in each
path, so cross-sectional dependence (strategy vs. benchmark vs. SPY on the same
day) is preserved. Index sequences depend only on (seed, n_obs, horizon,
mean_block, path number), never on the data values: two strategies measured on
the same sample length draw identical dates (common random numbers).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def stationary_bootstrap_indices(n_obs: int, n_days_out: int, mean_block: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """One path of resampled row indices.

    At each step a new block starts with probability p = 1 / mean_block (block
    lengths are geometric with mean `mean_block`); otherwise the previous index
    advances by one, wrapping around the end of the sample. The first step
    always starts a block.
    """
    if n_obs < 1 or n_days_out < 1:
        raise ValueError("n_obs and n_days_out must be positive.")
    if mean_block < 1:
        raise ValueError("mean_block must be >= 1.")
    p = 1.0 / float(mean_block)
    new = rng.random(n_days_out) < p
    new[0] = True
    starts = rng.integers(0, n_obs, size=n_days_out)
    t = np.arange(n_days_out)
    last_start = np.maximum.accumulate(np.where(new, t, 0))
    return (starts[last_start] + (t - last_start)) % n_obs


def bootstrap_index_matrix(n_obs: int, n_paths: int, horizon_days: int,
                           mean_block_days: float, seed: int) -> np.ndarray:
    """(n_paths, horizon_days) index matrix. Path i uses its own child stream of
    SeedSequence([seed, n_obs, horizon_days, mean_block_days]), so path i is the
    same whether 300 or 1000 paths are requested."""
    ss = np.random.SeedSequence([int(seed), int(n_obs), int(horizon_days), int(round(mean_block_days))])
    out = np.empty((n_paths, horizon_days), dtype=np.int64)
    for i, child in enumerate(ss.spawn(n_paths)):
        out[i] = stationary_bootstrap_indices(n_obs, horizon_days, mean_block_days,
                                              np.random.default_rng(child))
    return out


def joint_paths(returns: pd.DataFrame, n_paths: int, horizon_days: int,
                mean_block_days: float, seed: int) -> np.ndarray:
    """Bootstrap all columns jointly: array of shape (n_paths, horizon_days, n_cols)."""
    values = np.asarray(returns.to_numpy(dtype=float))
    idx = bootstrap_index_matrix(len(values), n_paths, horizon_days, mean_block_days, seed)
    return values[idx]
