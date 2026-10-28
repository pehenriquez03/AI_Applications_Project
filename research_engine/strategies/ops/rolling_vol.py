"""Total volatility from weekly returns (Blitz-van Vliet style low volatility)."""
from __future__ import annotations

import warnings

import numpy as np

from research_engine.strategies.registry import register
from ._util import cal_pos, empty


@register("rolling_vol")
def rolling_vol(ctx, dates, *, window_days: int = 756, freq: str = "W-FRI", min_obs: int = 100):
    """Std (ddof=1, not annualized) of simple weekly returns from week-ending closes (adj_close,
    last close of each W-FRI week) over weeks that END within the trailing `window_days`
    trading days up to `date`. Only completed weeks (label <= date) are used. NaN if fewer
    than `min_obs` weekly returns."""
    W = ctx.adj_close.resample(freq).last()
    WR = (W / W.shift(1) - 1.0)
    WR = WR.where(W.notna() & W.shift(1).notna())
    labels = WR.index
    A = WR.to_numpy()
    out = empty(ctx, dates)
    vals = out.to_numpy().copy()
    for k, d in enumerate(out.index):
        e = cal_pos(ctx, d)
        start = ctx.calendar[max(0, e - window_days + 1)]
        lo, hi = labels.searchsorted(start, side="left"), labels.searchsorted(d, side="right")
        block = A[lo:hi]
        n = np.sum(~np.isnan(block), axis=0)
        if block.shape[0] < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sd = np.nanstd(block, axis=0, ddof=1)
        vals[k] = np.where(n >= min_obs, sd, np.nan)
    out.iloc[:, :] = vals
    return out
