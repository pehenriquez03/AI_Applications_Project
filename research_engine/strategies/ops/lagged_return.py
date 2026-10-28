"""Seasonality (Heston-Sadka 2008): the return in the same calendar month one year earlier."""
from __future__ import annotations

from research_engine.strategies.registry import register
from ._util import anchor_month_end_pos, empty, month_end_prices


@register("lagged_month_return")
def lagged_month_return(ctx, dates, *, lag_months: int = 12):
    """Monthly return of the month `lag_months` months before the HOLDING month.

    Convention: `date` is the formation month-end of month m; the holding month is t = m+1.
    Heston-Sadka's lag-k return is the return of month t-k = m+1-k. With lag_months=12 this is
    the same calendar month as the holding month one year earlier, i.e. the month that ended
    lag_months-1 = 11 month-ends before `date` (formation 2015-01-30 -> February 2014 return).
    Monthly return = adj_close(end of that month) / adj_close(end of the month before) - 1.
    """
    P = month_end_prices(ctx)
    R = (P / P.shift(1) - 1.0).to_numpy()
    out = empty(ctx, dates)
    vals = out.to_numpy().copy()
    for k, d in enumerate(out.index):
        j = anchor_month_end_pos(ctx, d) - (lag_months - 1)
        if j < 1:
            continue
        vals[k] = R[j]
    out.iloc[:, :] = vals
    return out
