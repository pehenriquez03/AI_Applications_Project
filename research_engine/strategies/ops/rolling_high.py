"""52-week high (George-Hwang 2004): price / highest price over the past year."""
from __future__ import annotations

import warnings

import numpy as np

from research_engine.strategies.registry import register
from ._util import cal_pos, empty


@register("price_to_rolling_high")
def price_to_rolling_high(ctx, dates, *, window_days: int = 252, min_obs: int = 200,
                          high_field: str = "high"):
    """Split-only adjusted close at `date` / max split-only adjusted `high_field` over the
    trailing `window_days` trading days ending at `date` (inclusive).

    Splits are applied causally: prices in the window are rescaled by the splits that occur
    between them and `date` (ctx.split_factors), so no split after `date` is used. This is the
    same ratio as ctx.raw_split_adjusted() would give. NaN if the close at `date` is missing or
    fewer than `min_obs` highs are available."""
    G = np.cumprod(ctx.split_factors().to_numpy(), axis=0)
    C = ctx.close.to_numpy()
    H = getattr(ctx, high_field).to_numpy()
    out = empty(ctx, dates)
    vals = out.to_numpy().copy()
    for k, d in enumerate(out.index):
        e = cal_pos(ctx, d)
        s = max(0, e - window_days + 1)
        hw = H[s:e + 1] * G[s:e + 1] / G[e]
        n = np.sum(~np.isnan(hw), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            hmax = np.nanmax(hw, axis=0)
        vals[k] = np.where(n >= min_obs, C[e] / hmax, np.nan)
    out.iloc[:, :] = vals
    return out
