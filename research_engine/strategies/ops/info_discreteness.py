"""Frog in the pan (Da, Gurun, Warachka 2014): information discreteness."""
from __future__ import annotations

import numpy as np

from research_engine.strategies.registry import register
from ._util import anchor_month_end_pos, empty
from .cum_return import _cum


@register("info_discreteness")
def info_discreteness(ctx, dates, *, months: int = 12, skip_months: int = 1, min_days: int | None = None):
    """ID = sign(PRET) * (%negative days - %positive days) over the formation days.

    Formation window = holding-month-relative months t-12..t-2 (see pret_12_2): the daily returns
    after month-end m-12 through month-end m-1, where m is the month ending at `date`.
    Percentages use the stock's non-missing daily returns in the window (zero-return days count
    in the denominator only). NaN if fewer than `min_days` returns (default 15 per month)."""
    nmonths = months - skip_months
    min_days = 15 * nmonths if min_days is None else min_days
    pret = _cum(ctx, dates, nmonths, skip_months).to_numpy()
    R = ctx.returns.to_numpy()
    cal = ctx.calendar
    out = empty(ctx, dates)
    vals = out.to_numpy().copy()
    for k, d in enumerate(out.index):
        i = anchor_month_end_pos(ctx, d)
        if i - months < 0:
            continue
        s = cal.get_loc(ctx.month_ends[i - months]) + 1
        e = cal.get_loc(ctx.month_ends[i - skip_months])
        block = R[s:e + 1]
        n = np.sum(~np.isnan(block), axis=0)
        neg = np.sum(block < 0, axis=0)
        pos = np.sum(block > 0, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            idv = np.sign(pret[k]) * (neg - pos) / n
        vals[k] = np.where(n >= min_days, idv, np.nan)
    out.iloc[:, :] = vals
    return out
