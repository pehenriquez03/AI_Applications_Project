"""Cumulative return between two month-ends (Jegadeesh-Titman momentum)."""
from __future__ import annotations

import numpy as np

from research_engine.strategies.registry import register
from ._util import anchor_month_end_pos, empty, month_end_prices


def _cum(ctx, dates, months: int, skip_months: int):
    P = month_end_prices(ctx).to_numpy()
    out = empty(ctx, dates)
    vals = out.to_numpy().copy()
    for k, d in enumerate(out.index):
        i = anchor_month_end_pos(ctx, d)
        end, start = i - skip_months, i - skip_months - months
        if start < 0:
            continue
        vals[k] = P[end] / P[start] - 1.0
    out.iloc[:, :] = vals
    return out


@register("cum_return")
def cum_return(ctx, dates, *, months: int = 6, skip_months: int = 0):
    """adj_close at month-end t-skip / adj_close at month-end t-skip-months - 1, where t is the
    last month-end <= date. NaN if either price is missing on its month-end."""
    return _cum(ctx, dates, months, skip_months)


@register("pret_12_2")
def pret_12_2(ctx, dates, *, months: int = 12, skip_months: int = 1):
    """Past return over months t-12..t-2 relative to the holding month t (= the month after
    `date`): formation month m = t-1 is skipped, so PRET = P(end m-1) / P(end m-12) - 1.
    Equals cum_return(months=months-skip_months, skip_months=skip_months)."""
    return _cum(ctx, dates, months - skip_months, skip_months)
