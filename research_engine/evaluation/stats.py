"""Statistics used by the scorecards: Newey-West, annualization, regressions.

All HAC estimators use Bartlett weights w_j = 1 - j/(L+1) and no small-sample
correction (matches statsmodels cov_type="HAC", use_correction=False).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

TRADING_DAYS = 252


def _hac_cov(X: np.ndarray, u: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West covariance of OLS coefficients: (X'X)^-1 S (X'X)^-1."""
    n = X.shape[0]
    lags = int(max(0, min(lags, n - 1)))
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1.0)
        G = Xu[j:].T @ Xu[:-j]
        S = S + w * (G + G.T)
    XtX_inv = np.linalg.pinv(X.T @ X)
    return XtX_inv @ S @ XtX_inv


def newey_west_tstat(x: pd.Series, lags: int) -> tuple[float, float, float]:
    """(mean, t, se) for the mean of a series, Newey-West (Bartlett) standard error."""
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else float("nan"), float("nan"), float("nan"))
    v = x.to_numpy(float)
    mean = float(v.mean())
    cov = _hac_cov(np.ones((n, 1)), v - mean, lags)
    se = float(np.sqrt(max(cov[0, 0], 0.0)))
    t = mean / se if se > 0 else float("nan")
    return mean, t, se


def monthly_from_daily(r: pd.Series) -> pd.Series:
    """Compound daily returns within each calendar month. Index = month-end Timestamp."""
    r = pd.Series(r).dropna()
    if r.empty:
        return r
    g = (1.0 + r).groupby(r.index.to_period("M")).prod() - 1.0
    g.index = g.index.to_timestamp(how="end").normalize()
    return g


def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1.0 + pd.Series(r).fillna(0.0)).cumprod()
    peak = np.maximum.accumulate(np.maximum(wealth.to_numpy(), 1.0))
    return pd.Series(wealth.to_numpy() / peak - 1.0, index=wealth.index)


def annualize(r: pd.Series, rf: pd.Series | None = None) -> dict:
    """Performance of a daily return series.

    cagr = end_wealth ** (252/n) - 1; vol = std(ddof=1) * sqrt(252);
    sharpe = mean(r - rf) * 252 / vol (rf = 0 when not given); max_drawdown is
    the most negative peak-to-trough change of wealth (starting at 1).
    """
    r = pd.Series(r).dropna()
    n = len(r)
    if n < 2:
        nan = float("nan")
        return dict(cagr=nan, vol=nan, sharpe=nan, max_drawdown=nan, end_wealth=nan,
                    n_days=n, drawdown=pd.Series(dtype=float))
    end_wealth = float((1.0 + r).prod())
    cagr = end_wealth ** (TRADING_DAYS / n) - 1.0 if end_wealth > 0 else -1.0
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    if rf is not None:
        ex = r - pd.Series(rf).reindex(r.index).fillna(0.0)
    else:
        ex = r
    sharpe = float(ex.mean() * TRADING_DAYS / vol) if vol > 0 else float("nan")
    dd = drawdown_series(r)
    return dict(cagr=float(cagr), vol=vol, sharpe=sharpe, max_drawdown=float(dd.min()),
                end_wealth=end_wealth, n_days=n, drawdown=dd)


def spearman_monotonicity(group_means: list) -> tuple[float, float]:
    """Spearman rho (and p) between group index 1..K and the group mean returns."""
    y = np.asarray(group_means, float)
    if len(y) < 3 or np.isnan(y).any() or np.ptp(y) == 0:
        return float("nan"), float("nan")
    res = sps.spearmanr(np.arange(1, len(y) + 1), y)
    return round(float(res[0]), 12), float(res[1])


def alpha_regression(y_daily_excess: pd.Series, X_daily_factors: pd.DataFrame, lags: int,
                     periods_per_year: int = TRADING_DAYS) -> dict:
    """OLS of y on a constant and the factors with Newey-West standard errors.

    Returns alpha_annual (= daily intercept * 252), alpha_t (NW), betas by factor
    name, beta_t, r2, n_obs.
    """
    df = pd.concat([pd.Series(y_daily_excess).rename("__y__"), X_daily_factors], axis=1, join="inner").dropna()
    n = len(df)
    k = X_daily_factors.shape[1]
    if n <= k + 2:
        return dict(alpha_annual=float("nan"), alpha_t=float("nan"), betas={c: float("nan") for c in X_daily_factors.columns},
                    beta_t={c: float("nan") for c in X_daily_factors.columns}, r2=float("nan"), n_obs=n)
    y = df["__y__"].to_numpy(float)
    X = np.column_stack([np.ones(n), df[list(X_daily_factors.columns)].to_numpy(float)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ coef
    cov = _hac_cov(X, u, lags)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    t = np.divide(coef, se, out=np.full_like(coef, np.nan), where=se > 0)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((u ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    cols = list(X_daily_factors.columns)
    return dict(alpha_annual=float(coef[0] * periods_per_year), alpha_t=float(t[0]),
                alpha_daily=float(coef[0]),
                betas={c: float(coef[i + 1]) for i, c in enumerate(cols)},
                beta_t={c: float(t[i + 1]) for i, c in enumerate(cols)},
                r2=float(r2), n_obs=int(n))
