"""avg_lagged_month_return and oil_beta: hand checks, NaN rules, registration, no look-ahead."""
import numpy as np
import pandas as pd
import pytest

from research_engine.contracts import OIL_PRICES, SNAPSHOT_DIR
from research_engine.data.loaders import DataContext
from research_engine.strategies import registry
from research_engine.strategies.ops import oil_beta as oil_mod

T = pd.Timestamp


@pytest.fixture(scope="module")
def ctx():
    return DataContext.load()


@pytest.fixture(scope="module")
def adj():
    return pd.read_parquet(SNAPSHOT_DIR / "adj_close.parquet")


def _month_end(ctx, period):
    me = ctx.month_ends
    hit = me[me.to_period("M") == period]
    return hit[0] if len(hit) else None


def _month_return(ctx, adj, ticker, period):
    """Return of calendar month `period` from the raw parquet (None if unavailable)."""
    end, prev = _month_end(ctx, period), _month_end(ctx, period - 1)
    if end is None or prev is None:
        return None
    return adj.at[end, ticker] / adj.at[prev, ticker] - 1.0


def test_registered():
    assert registry.get("avg_lagged_month_return").op_name == "avg_lagged_month_return"
    assert registry.get("oil_beta").op_name == "oil_beta"
    assert {"avg_lagged_month_return", "oil_beta"} <= set(registry.list_ops())


def test_avg_lagged_hand_check(ctx, adj):
    d = T("2018-11-30")                               # holding month = December 2018
    hold = d.to_period("M") + 1
    sig = registry.compute(ctx, pd.DatetimeIndex([d]), "avg_lagged_month_return")
    for t in ["MSFT", "XOM", "AAPL"]:
        rets = [_month_return(ctx, adj, t, hold - L) for L in (24, 36, 48, 60)]
        assert all(r is not None for r in rets)
        assert abs(sig.at[d, t] - np.mean(rets)) < 1e-10
    # lag 12 convention check against lagged_month_return: same calendar month as the holding month
    one = registry.compute(ctx, pd.DatetimeIndex([d]), "avg_lagged_month_return", {"lags": (12,), "min_lags": 1})
    assert abs(one.at[d, "MSFT"] - _month_return(ctx, adj, "MSFT", T("2017-12-31").to_period("M"))) < 1e-10
    # custom lags
    two = registry.compute(ctx, pd.DatetimeIndex([d]), "avg_lagged_month_return", {"lags": (12, 24), "min_lags": 2})
    exp = np.mean([_month_return(ctx, adj, "XOM", hold - L) for L in (12, 24)])
    assert abs(two.at[d, "XOM"] - exp) < 1e-10


def test_avg_lagged_nan_before_enough_lags(ctx, adj):
    # snapshot starts 2012-01: holding 2013-07 needs 2011-07 for lag 24 -> nothing available
    early = registry.compute(ctx, pd.DatetimeIndex([T("2013-06-28")]), "avg_lagged_month_return")
    assert early.isna().all().all()
    # 2014-12 formation (hold 2015-01): lag 24 = Jan 2013 exists, lag 36 = Jan 2012 lacks a prior month-end
    one_lag = registry.compute(ctx, pd.DatetimeIndex([T("2014-12-31")]), "avg_lagged_month_return")
    assert one_lag.isna().all().all()
    relaxed = registry.compute(ctx, pd.DatetimeIndex([T("2014-12-31")]), "avg_lagged_month_return", {"min_lags": 1})
    assert abs(relaxed.at[T("2014-12-31"), "MSFT"] - _month_return(ctx, adj, "MSFT", T("2013-01-31").to_period("M"))) < 1e-10
    # 2015-01 formation (hold 2015-02): lags 24 and 36 exist, 48 and 60 do not
    d = T("2015-01-30")
    two = registry.compute(ctx, pd.DatetimeIndex([d]), "avg_lagged_month_return")
    hold = d.to_period("M") + 1
    exp = np.mean([_month_return(ctx, adj, "MSFT", hold - L) for L in (24, 36)])
    assert abs(two.at[d, "MSFT"] - exp) < 1e-10
    strict = registry.compute(ctx, pd.DatetimeIndex([d]), "avg_lagged_month_return", {"min_lags": 3})
    assert strict.isna().all().all()


def _oil_changes_by_month():
    df = pd.read_csv(OIL_PRICES)
    lv = pd.Series(df["Factor Level"].to_numpy(), index=pd.to_datetime(df["Date"], format="%m/%d/%Y")).sort_index()
    ch = lv.pct_change()
    ch.index = ch.index.to_period("M")
    return ch


def test_oil_beta_hand_check(ctx, adj):
    d = T("2018-11-30")
    sig = registry.compute(ctx, pd.DatetimeIndex([d]), "oil_beta")
    ch = _oil_changes_by_month()
    periods = [d.to_period("M") - i for i in range(35, -1, -1)]          # 36 months ending Nov 2018
    x = np.array([ch[p] for p in periods])
    for t in ["MSFT", "XOM", "AAPL"]:
        y = np.array([_month_return(ctx, adj, t, p) for p in periods])
        assert abs(sig.at[d, t] - np.polyfit(x, y, 1)[0]) < 1e-8
    # a different window
    s12 = registry.compute(ctx, pd.DatetimeIndex([d]), "oil_beta", {"window_months": 12, "min_months": 12})
    x12, y12 = x[-12:], np.array([_month_return(ctx, adj, "XOM", p) for p in periods[-12:]])
    assert abs(s12.at[d, "XOM"] - np.polyfit(x12, y12, 1)[0]) < 1e-8


def test_oil_beta_nan_short_history(ctx):
    # only 11 monthly returns exist by 2012-12
    early = registry.compute(ctx, pd.DatetimeIndex([T("2012-12-31")]), "oil_beta")
    assert early.isna().all().all()
    # exactly 24 returns (Feb 2012 .. Jan 2014): defined at min_months=24, NaN at 25
    d = T("2014-01-31")
    ok = registry.compute(ctx, pd.DatetimeIndex([d]), "oil_beta")
    assert np.isfinite(ok.at[d, "MSFT"])
    strict = registry.compute(ctx, pd.DatetimeIndex([d]), "oil_beta", {"min_months": 25})
    assert strict.isna().all().all()
    # a stock whose history starts 2021-02 (synthetic: MSFT blanked before it): < 24 returns by mid-2021
    adj2 = ctx.adj_close.copy()
    adj2.loc[adj2.index < "2021-02-01", "MSFT"] = np.nan
    c2 = ctx.replace_prices(adj_close=adj2)
    d2 = T("2021-06-30")
    late = registry.compute(c2, pd.DatetimeIndex([d2]), "oil_beta")
    assert np.isnan(late.at[d2, "MSFT"]) and np.isfinite(late.at[d2, "AAPL"])


def test_oil_alignment_no_stale_values(ctx):
    me = pd.DatetimeIndex([T("2024-03-28"), T("2024-04-30"), T("2025-06-30")])
    x = oil_mod.oil_changes_on(me, oil_mod.read_oil_levels())
    assert np.isfinite(x[0]) and np.isnan(x[1]) and np.isnan(x[2])       # file ends with the 2024-03 level


def _perturbed(ctx, after, seed=7):
    rng = np.random.default_rng(seed)
    mask = ctx.calendar > after
    out = {}
    for f in ("adj_close", "close", "high", "low"):
        p = getattr(ctx, f).copy()
        noise = 1 + 0.05 * rng.standard_normal((int(mask.sum()), p.shape[1]))
        p.loc[mask] = p.loc[mask].to_numpy() * noise
        out[f] = p
    return ctx.replace_prices(**out)


@pytest.mark.parametrize("op", ["avg_lagged_month_return", "oil_beta"])
def test_no_look_ahead_prices(ctx, op):
    D = T("2016-06-15")
    c2 = _perturbed(ctx, D)
    dates = ctx.month_ends[(ctx.month_ends >= "2014-01-01") & (ctx.month_ends <= "2019-12-31")].append(
        pd.DatetimeIndex([D])).sort_values()
    a, b = registry.compute(ctx, dates, op), registry.compute(c2, dates, op)
    before = dates <= D
    assert a.loc[before].notna().any().any()
    pd.testing.assert_frame_equal(a.loc[before], b.loc[before], check_exact=True)
    assert not a.loc[~before].equals(b.loc[~before])


def test_no_look_ahead_oil(ctx, monkeypatch):
    D = T("2018-05-31")
    real = oil_mod.read_oil_levels()
    fake = real.copy()
    fake[fake.index > D] *= np.random.default_rng(3).uniform(0.5, 1.5, int((fake.index > D).sum()))
    dates = ctx.month_ends[(ctx.month_ends >= "2016-01-01") & (ctx.month_ends <= "2019-12-31")]
    a = registry.compute(ctx, dates, "oil_beta")
    monkeypatch.setattr(oil_mod, "read_oil_levels", lambda *args, **kw: fake)
    b = registry.compute(ctx, dates, "oil_beta")
    before = dates <= D
    pd.testing.assert_frame_equal(a.loc[before], b.loc[before], check_exact=True)
    assert not a.loc[~before].equals(b.loc[~before])
