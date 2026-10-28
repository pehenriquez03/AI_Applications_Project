"""MAX (Bali, Cakici, Whitelaw 2011): largest daily return in the formation month."""
from __future__ import annotations

import warnings

import numpy as np

from research_engine.contracts import MIN_FORMATION_DAYS
from research_engine.strategies.registry import register
from ._util import cal_pos, empty, month_start_pos


@register("rolling_max_daily_return")
def rolling_max_daily_return(ctx, dates, *, window_months: int = 1, min_days: int = MIN_FORMATION_DAYS):
    """Max daily simple return (adj_close) over the `window_months` calendar months ending
    with the month containing `date`, using only days <= date. NaN with < min_days returns."""
    R = ctx.returns.to_numpy()
    out = empty(ctx, dates)
    vals = out.to_numpy().copy()
    for k, d in enumerate(out.index):
        e, s = cal_pos(ctx, d), month_start_pos(ctx, d, window_months)
        if e < s:
            continue
        block = R[s:e + 1]
        n = np.sum(~np.isnan(block), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mx = np.nanmax(block, axis=0)
        vals[k] = np.where(n >= min_days, mx, np.nan)
    out.iloc[:, :] = vals
    return out
