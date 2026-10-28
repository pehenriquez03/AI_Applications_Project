"""Average of annual-lag monthly returns (Heston-Sadka 2008 style seasonality signal)."""
from __future__ import annotations

import numpy as np

from research_engine.strategies.registry import register
from .lagged_return import lagged_month_return
from ._util import empty


@register("avg_lagged_month_return")
def avg_lagged_month_return(ctx, dates, *, lags=(24, 36, 48, 60), min_lags: int = 2):
    """Mean over `lags` L of the monthly return of the month L months before the HOLDING month.

    Convention (identical to `lagged_month_return`, which this op calls for every lag): `date`
    is the formation month-end of month m and the holding month is t = m+1; lag L is the return
    of month t-L = m+1-L, i.e. the month that ended L-1 month-ends before `date`. Lag 12 is the
    same calendar month as the holding month one year earlier (formation 2015-01-30 -> February
    2014). Monthly return = adj_close(month-end) / adj_close(previous month-end) - 1.

    A lag is "available" for a stock when its monthly return is finite (the month exists in the
    snapshot and the stock priced on both month-ends). The signal is the mean of the available
    lags and NaN when fewer than `min_lags` are available. Uses only data with index <= date.
    """
    lags = tuple(int(L) for L in lags)
    out = empty(ctx, dates)
    if not lags:
        return out
    stack = np.stack([lagged_month_return(ctx, dates, lag_months=L).to_numpy() for L in lags])
    ok = np.isfinite(stack)
    n = ok.sum(axis=0)
    total = np.where(ok, stack, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / n
    out.iloc[:, :] = np.where(n >= max(int(min_lags), 1), mean, np.nan)
    return out
