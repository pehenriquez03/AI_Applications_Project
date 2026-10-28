"""Shared helpers for ops: month arithmetic on the trading calendar (no look-ahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def anchor_month_end_pos(ctx, date) -> int:
    """Position in ctx.month_ends of the last month-end <= date (-1 if none)."""
    return int(ctx.month_ends.searchsorted(pd.Timestamp(date), side="right")) - 1


def month_start_pos(ctx, date, window_months: int) -> int:
    """Calendar position of the first trading day of the window of `window_months`
    calendar months ending with the month that contains `date`."""
    p = pd.Timestamp(date).to_period("M") - (window_months - 1)
    start = p.to_timestamp(how="start")
    return int(ctx.calendar.searchsorted(start, side="left"))


def cal_pos(ctx, date) -> int:
    """Calendar position of the last trading day <= date."""
    return int(ctx.calendar.searchsorted(pd.Timestamp(date), side="right")) - 1


def empty(ctx, dates) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=pd.DatetimeIndex(dates, name="date"), columns=ctx.tickers)


def month_end_prices(ctx) -> pd.DataFrame:
    """adj_close on each month-end (exact row; NaN if the stock has no price that day)."""
    return ctx.adj_close.loc[ctx.month_ends]
