"""Idiosyncratic volatility (Ang, Hodrick, Xing, Zhang 2006)."""
from __future__ import annotations

import numpy as np

from research_engine.contracts import MIN_FORMATION_DAYS
from research_engine.strategies.registry import register
from ._util import cal_pos, empty, month_start_pos


@register("residual_vol")
def residual_vol(ctx, dates, *, window_months: int = 1, factors=("mkt_rf", "smb", "hml"),
                 min_days: int = MIN_FORMATION_DAYS, excess: bool = True):
    """Std (ddof=1) of daily residuals from OLS of the stock's (excess, if `excess`) daily return
    on an intercept and `factors`, over the days of the `window_months` calendar months ending
    with the month containing `date` (days <= date only). NaN with < min_days observations."""
    R = ctx.returns.to_numpy()
    aliases = {"ff3": ("mkt_rf", "smb", "hml"), "ff4": ("mkt_rf", "smb", "hml", "mom"), "capm": ("mkt_rf",), "market": ("mkt_rf",)}
    if isinstance(factors, str):
        factors = aliases.get(factors.lower(), (factors,))
    F = ctx.factors[list(factors)].to_numpy()
    rf = ctx.factors["rf"].to_numpy()
    out = empty(ctx, dates)
    vals = out.to_numpy().copy()
    for k, d in enumerate(out.index):
        e, s = cal_pos(ctx, d), month_start_pos(ctx, d, window_months)
        if e < s:
            continue
        Y = R[s:e + 1] - (rf[s:e + 1, None] if excess else 0.0)
        X = np.column_stack([np.ones(e - s + 1), F[s:e + 1]])
        ok = ~np.isnan(Y)
        n = ok.sum(axis=0)
        res = np.full(Y.shape[1], np.nan)
        full = n == Y.shape[0]
        if full.any() and Y.shape[0] > X.shape[1]:
            b, *_ = np.linalg.lstsq(X, Y[:, full], rcond=None)
            res[full] = np.std(Y[:, full] - X @ b, axis=0, ddof=1)
        for c in np.where(~full & (n >= min_days))[0]:
            m = ok[:, c]
            b, *_ = np.linalg.lstsq(X[m], Y[m, c], rcond=None)
            res[c] = np.std(Y[m, c] - X[m] @ b, ddof=1)
        res[n < min_days] = np.nan
        vals[k] = res
    out.iloc[:, :] = vals
    return out
