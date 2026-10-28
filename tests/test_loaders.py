"""DataContext: hand-checked values, calendar, eligibility, split rebuild."""
import numpy as np
import pandas as pd
import pytest

from research_engine.contracts import SNAPSHOT_DIR
from research_engine.data.loaders import DataContext

T = pd.Timestamp


@pytest.fixture(scope="module")
def ctx():
    return DataContext.load(exclude=())  # raw snapshot: these tests check the 415-name load


@pytest.fixture(scope="module")
def raw():
    return {f: pd.read_parquet(SNAPSHOT_DIR / f"{f}.parquet") for f in ("adj_close", "close", "high", "ff3_daily")}


def test_hand_checked_values(ctx, raw):
    a = raw["adj_close"]
    # 1. AAPL adjusted close on the 4:1 split day
    assert ctx.adj_close.at[T("2020-08-31"), "AAPL"] == a.at[T("2020-08-31"), "AAPL"]
    assert abs(ctx.adj_close.at[T("2020-08-31"), "AAPL"] - 125.05754089355469) < 1e-9
    # 2. MSFT daily simple return on 2016-03-15
    exp = a.at[T("2016-03-15"), "MSFT"] / a.at[T("2016-03-14"), "MSFT"] - 1
    assert abs(ctx.returns.at[T("2016-03-15"), "MSFT"] - exp) < 1e-15
    assert abs(exp - 0.007899174813155385) < 1e-12
    # 3. Fama-French market excess return on 2019-06-28 (decimal)
    assert ctx.factors.at[T("2019-06-28"), "mkt_rf"] == raw["ff3_daily"].at[T("2019-06-28"), "Mkt-RF"]
    assert abs(ctx.factors.at[T("2019-06-28"), "mkt_rf"] - 0.0068) < 1e-12


def test_spy_split_out(ctx, raw):
    assert "SPY" not in ctx.tickers and "SPY" not in ctx.adj_close.columns
    assert len(ctx.tickers) == 415
    assert ctx.spy_adj_close.equals(raw["adj_close"]["SPY"].set_axis(ctx.calendar).rename("SPY")) or \
        np.allclose(ctx.spy_adj_close.to_numpy(), raw["adj_close"]["SPY"].to_numpy())
    assert abs(ctx.spy_returns.iloc[5] - (ctx.spy_adj_close.iloc[5] / ctx.spy_adj_close.iloc[4] - 1)) < 1e-15
    assert list(ctx.factors.columns) == ["mkt_rf", "smb", "hml", "mom", "rf"]


def test_calendar_nyse_like(ctx):
    cal = ctx.calendar
    assert cal.is_monotonic_increasing and cal.is_unique
    assert (cal.day_of_week < 5).all()
    per_year = pd.Series(1, index=cal).groupby(cal.year).sum()
    assert per_year.loc[2013:2025].between(248, 254).all()
    assert T("2015-12-25") not in cal and T("2019-07-04") not in cal
    assert len(ctx.month_ends) == len(ctx.first_days) == 14 * 12
    nxt = pd.DatetimeIndex([ctx.execution_day(d) for d in ctx.month_ends[:-1]])
    assert nxt.equals(ctx.first_days[1:])
    assert (ctx.month_ends.to_period("M") == ctx.first_days.to_period("M")).all()


def test_eligible_count_rises(ctx):
    counts = [len(ctx.eligible(d)) for d in ctx.month_ends[ctx.month_ends.month == 12]]
    assert counts[0] == 0                       # 2012: nobody has 252 days yet
    assert counts[1] < counts[4] < counts[8] < counts[-1]
    assert all(b >= a - 2 for a, b in zip(counts, counts[1:]))
    assert counts[-1] == 415


def test_add_date_masking(ctx):
    assert ctx.add_dates["SMCI"] == T("2024-03-18")
    assert ctx.adj_close["SMCI"].loc[:"2023-12-29"].notna().sum() > 252
    assert "SMCI" not in ctx.eligible("2023-12-29")
    assert "SMCI" not in ctx.eligible("2024-02-29")
    assert "SMCI" in ctx.eligible("2024-03-28")
    # NaT add date means always eligible (given history)
    c3 = ctx.replace_prices()
    c3.add_dates.loc["SMCI"] = pd.NaT
    c3 = c3.replace_prices()  # rebuild arrays from the edited add_dates
    assert "SMCI" in c3.eligible("2023-12-29")


def test_history_and_price_floor(ctx):
    first = ctx.adj_close["PARA"].first_valid_index()
    assert first == T("2021-02-12")
    n = ctx.adj_close["PARA"].loc[:"2021-12-31"].notna().sum()
    assert n < 252 and "PARA" not in ctx.eligible("2021-12-31")
    assert len(ctx.eligible("2020-06-30", price_floor=50.0)) < len(ctx.eligible("2020-06-30"))
    low = ctx.close.loc[T("2020-06-30"), ctx.eligible("2020-06-30")]
    assert set(ctx.eligible("2020-06-30", price_floor=50.0)) == set(low[low >= 50.0].index)


def test_split_rebuild_aapl(ctx):
    # The snapshot's close is already split-adjusted by Yahoo, so nothing is applied ...
    rs = ctx.raw_split_adjusted("close")["AAPL"]
    jump = rs.loc[T("2020-08-31")] / rs.loc[T("2020-08-28")]
    assert 0.9 < jump < 1.1
    # ... and on a genuinely raw series (pre-split prices x4) the rebuild removes the 4:1 drop.
    pre = ctx.calendar < T("2020-08-31")
    fake = {f: getattr(ctx, f).copy() for f in ("close", "high", "low")}
    for f in fake:
        fake[f].loc[pre, "AAPL"] *= 4.0
    c2 = ctx.replace_prices(**fake)
    raw_jump = c2.close.at[T("2020-08-31"), "AAPL"] / c2.close.at[T("2020-08-28"), "AAPL"]
    assert raw_jump < 0.3
    for f in ("close", "high", "low"):
        rebuilt = c2.raw_split_adjusted(f)["AAPL"]
        assert np.allclose(rebuilt.to_numpy(), getattr(ctx, f)["AAPL"].to_numpy(), rtol=1e-12, equal_nan=True)
    assert c2.split_factors().at[T("2020-08-31"), "AAPL"] == 4.0
    assert c2.split_factors().at[T("2014-06-09"), "AAPL"] == 1.0   # 7:1 split already folded in
