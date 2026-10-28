"""Oil beta: slope of monthly stock returns on monthly oil price changes (trailing OLS)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research_engine.contracts import OIL_PRICES
from research_engine.strategies.registry import register
from ._util import anchor_month_end_pos, empty, month_end_prices


def read_oil_levels(path=OIL_PRICES) -> pd.Series:
    """Monthly oil levels from the CSV (columns Date, "Factor Level"), parsed and sorted by date."""
    df = pd.read_csv(path)
    s = pd.Series(df["Factor Level"].to_numpy(dtype=float),
                  index=pd.to_datetime(df["Date"], format="%m/%d/%Y"), name="oil")
    return s.sort_index()


def oil_changes_on(month_ends: pd.DatetimeIndex, levels: pd.Series) -> np.ndarray:
    """Simple percentage oil change aligned to each month-end (NaN where unavailable).

    Alignment rule: changes are computed on the original (sorted) monthly series, then each
    month-end takes the change of the LAST observation dated at or before it. The observation
    must lie in the same calendar month as the month-end, otherwise the value is NaN; this
    stops a stale change being repeated past the end of the oil file (or over a gap). The CSV
    dates each month at its first day, so the observation used at month-end m is that month's.
    """
    chg = levels.pct_change()
    pos = chg.index.searchsorted(month_ends, side="right") - 1
    ok = pos >= 0
    pos = np.where(ok, pos, 0)
    same_month = chg.index[pos].to_period("M") == month_ends.to_period("M")
    return np.where(ok & same_month, chg.to_numpy()[pos], np.nan)


@register("oil_beta")
def oil_beta(ctx, dates, *, window_months: int = 36, min_months: int = 24):
    """OLS slope of each stock's monthly return on the monthly oil price change.

    Window: the `window_months` month-ends ending at the formation month (the last month-end
    <= date). Stock return = adj_close month-end / previous month-end - 1; oil change as in
    `oil_changes_on`. A month enters a stock's regression only when both are finite; the slope
    is NaN with fewer than `min_months` such months (or a constant oil series). Only oil
    observations dated <= the formation month-end and prices with index <= date are used.
    Vectorised over tickers: slope = sum((x-mx)(y-my)) / sum((x-mx)^2) per column.
    """
    P = month_end_prices(ctx)
    R = (P / P.shift(1) - 1.0).to_numpy()
    X = oil_changes_on(ctx.month_ends, read_oil_levels())
    out = empty(ctx, dates)
    vals = out.to_numpy().copy()
    min_months = max(int(min_months), 2)
    for k, d in enumerate(out.index):
        j = anchor_month_end_pos(ctx, d)
        if j < 1:
            continue
        lo = max(1, j - window_months + 1)
        y, x = R[lo:j + 1], X[lo:j + 1]
        m = np.isfinite(y) & np.isfinite(x)[:, None]
        n = m.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            xm = np.where(m, x[:, None], 0.0)
            ym = np.where(m, y, 0.0)
            dx = np.where(m, xm - (xm.sum(axis=0) / n), 0.0)
            dy = np.where(m, ym - (ym.sum(axis=0) / n), 0.0)
            sxx = (dx * dx).sum(axis=0)
            slope = (dx * dy).sum(axis=0) / sxx
        vals[k] = np.where((n >= min_months) & (sxx > 0), slope, np.nan)
    out.iloc[:, :] = vals
    return out
