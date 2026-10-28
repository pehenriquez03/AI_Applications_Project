"""Signal ops: hand checks from the raw parquet, conventions, and a look-ahead mutation test."""
import numpy as np
import pandas as pd
import pytest

from research_engine.contracts import SNAPSHOT_DIR
from research_engine.data.loaders import DataContext
from research_engine.strategies import registry

T = pd.Timestamp
TICKERS = ["AAPL", "MSFT", "XOM"]          # AAPL: 4:1 split on 2020-08-31
DATES = pd.DatetimeIndex([T("2016-03-31"), T("2020-09-30")])


@pytest.fixture(scope="module")
def ctx():
    return DataContext.load(exclude=())  # raw snapshot incl. PARA for the min-days case


@pytest.fixture(scope="module")
def adj():
    return pd.read_parquet(SNAPSHOT_DIR / "adj_close.parquet")


def test_registry():
    ops = registry.list_ops()
    for name in ["rolling_max_daily_return", "cum_return", "lagged_month_return", "residual_vol",
                 "rolling_vol", "price_to_rolling_high", "pret_12_2", "info_discreteness"]:
        assert name in ops
    with pytest.raises(KeyError):
        registry.get("no_such_op")


def test_max_hand_check(ctx, adj):
    dates = pd.DatetimeIndex([T("2016-03-31"), T("2020-08-31")])      # Aug 2020 contains the AAPL split
    sig = registry.compute(ctx, dates, "rolling_max_daily_return")
    for d in dates:
        r = adj[TICKERS].pct_change(fill_method=None)
        month = r[(r.index.year == d.year) & (r.index.month == d.month)]
        for t in TICKERS:
            assert abs(sig.at[d, t] - month[t].max()) < 1e-10


def test_max_min_days(ctx):
    sig = registry.compute(ctx, pd.DatetimeIndex([T("2021-02-26")]), "rolling_max_daily_return")
    assert np.isnan(sig.at[T("2021-02-26"), "PARA"])                  # first price 2021-02-12: < 15 days


def test_cum_return_hand_check(ctx, adj):
    me = ctx.month_ends
    sig6 = registry.compute(ctx, DATES, "cum_return", {"months": 6, "skip_months": 0})
    sig12 = registry.compute(ctx, DATES, "cum_return", {"months": 11, "skip_months": 1})
    pret = registry.compute(ctx, DATES, "pret_12_2")
    for d in DATES:
        i = me.get_loc(d)
        for t in TICKERS:
            assert abs(sig6.at[d, t] - (adj.at[me[i], t] / adj.at[me[i - 6], t] - 1)) < 1e-10
            assert abs(sig12.at[d, t] - (adj.at[me[i - 1], t] / adj.at[me[i - 12], t] - 1)) < 1e-10
            assert abs(pret.at[d, t] - sig12.at[d, t]) < 1e-15


def test_price_to_rolling_high_hand_check(ctx):
    close = pd.read_parquet(SNAPSHOT_DIR / "close.parquet")
    high = pd.read_parquet(SNAPSHOT_DIR / "high.parquet")
    sig = registry.compute(ctx, DATES, "price_to_rolling_high")
    for d in DATES:
        i = close.index.get_loc(d)
        for t in TICKERS:
            exp = close[t].iloc[i] / high[t].iloc[i - 251:i + 1].max()
            assert abs(sig.at[d, t] - exp) < 1e-10
    # The window ending 2020-09-30 spans the AAPL split; Yahoo's close/high are split-adjusted,
    # so no 4:1 step is in the window ...
    w = close["AAPL"].loc["2019-10-01":"2020-09-30"]
    assert (w / w.shift(1)).min() > 0.8
    # ... and on a truly raw series (pre-split prices x4) the op gives the same value.
    pre = ctx.calendar < T("2020-08-31")
    fake = {f: getattr(ctx, f).copy() for f in ("close", "high", "low")}
    for f in fake:
        fake[f].loc[pre, "AAPL"] *= 4.0
    sig2 = registry.compute(ctx.replace_prices(**fake), DATES, "price_to_rolling_high")
    assert abs(sig2.at[T("2020-09-30"), "AAPL"] - sig.at[T("2020-09-30"), "AAPL"]) < 1e-10


def test_lagged_month_return_convention(ctx, adj):
    # formation 2015-01-30 -> holding month Feb 2015 -> signal = Feb 2014 return
    d = T("2015-01-30")
    sig = registry.compute(ctx, pd.DatetimeIndex([d]), "lagged_month_return")
    exp = adj.at[T("2014-02-28"), "MSFT"] / adj.at[T("2014-01-31"), "MSFT"] - 1
    assert abs(sig.at[d, "MSFT"] - exp) < 1e-10


def test_residual_vol_matches_lstsq(ctx):
    d = T("2019-05-31")
    sig = registry.compute(ctx, pd.DatetimeIndex([d]), "residual_vol")
    m = (ctx.calendar.year == 2019) & (ctx.calendar.month == 5)
    y = ctx.returns.loc[m, "MSFT"] - ctx.factors.loc[m, "rf"]
    X = np.column_stack([np.ones(m.sum()), ctx.factors.loc[m, ["mkt_rf", "smb", "hml"]]])
    b = np.linalg.lstsq(X, y.to_numpy(), rcond=None)[0]
    assert abs(sig.at[d, "MSFT"] - np.std(y.to_numpy() - X @ b, ddof=1)) < 1e-12


def test_rolling_vol_and_id(ctx):
    d = T("2018-12-31")
    vol = registry.compute(ctx, pd.DatetimeIndex([d]), "rolling_vol")
    wk = ctx.adj_close["MSFT"].loc[:d].resample("W-FRI").last()
    wr = (wk / wk.shift(1) - 1).loc[ctx.calendar[ctx.calendar.get_loc(d) - 755]:d].dropna()
    assert abs(vol.at[d, "MSFT"] - wr.std(ddof=1)) < 1e-12
    idv = registry.compute(ctx, pd.DatetimeIndex([d]), "info_discreteness")
    r = ctx.returns["MSFT"].loc["2018-01-01":"2018-11-30"]
    pret = registry.compute(ctx, pd.DatetimeIndex([d]), "pret_12_2").at[d, "MSFT"]
    exp = np.sign(pret) * ((r < 0).mean() - (r > 0).mean())
    assert abs(idv.at[d, "MSFT"] - exp) < 1e-12


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


@pytest.mark.parametrize("op", ["rolling_max_daily_return", "cum_return", "lagged_month_return", "residual_vol",
                                "rolling_vol", "price_to_rolling_high", "pret_12_2", "info_discreteness"])
def test_no_look_ahead(ctx, op):
    D = T("2018-06-15")
    c2 = _perturbed(ctx, D)
    dates = ctx.month_ends[(ctx.month_ends >= "2016-01-01") & (ctx.month_ends <= "2019-12-31")].append(
        pd.DatetimeIndex([D])).sort_values()
    a = registry.compute(ctx, dates, op)
    b = registry.compute(c2, dates, op)
    before = dates <= D
    pd.testing.assert_frame_equal(a.loc[before], b.loc[before], check_exact=True)
    assert not a.loc[~before].equals(b.loc[~before])           # the perturbation is visible after D
