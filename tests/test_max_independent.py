"""Independent reconciliation of the MAX lowest-decile strategy.

Note on what can match exactly: an equal-weight portfolio bought at a month-end and held to the
next month-end earns exactly the average of its members' month-end-to-month-end returns (the
weights drift, but the period return is the average). So the engine run with targets executed
at each formation month-end close and NO re-targeting inside the month must equal the
independent pure-pandas average to 1e-10. Daily re-targeting to constant weights compounds
sum(w*r) daily and differs from that by the within-month drift, which is bounded below.
"""
import json

import numpy as np
import pandas as pd
import pytest

from research_engine.backtest.engine import run_backtest
from research_engine.contracts import SNAPSHOT_DIR, UNIVERSE_FILE
from research_engine.data.loaders import DataContext
from research_engine.strategies.dev_run import max_spec
from research_engine.strategies.portfolios import build_targets


def independent_max_decile1(start="2015-01-01", end="2025-12-31"):
    """Pure pandas, from the files: {formation month-end: (set of names, next-month EW return)}."""
    adj = pd.read_parquet(SNAPSHOT_DIR / "adj_close.parquet")
    from research_engine.contracts import EXCLUDE_TICKERS
    adj = adj[[t for t in json.loads((SNAPSHOT_DIR / "manifest.json").read_text())["universe_tickers"] if t not in EXCLUDE_TICKERS]]
    csv = pd.read_csv(UNIVERSE_FILE)
    add = pd.Series(pd.to_datetime(csv["Date added"]).to_numpy(), index=csv["Symbol"].str.replace(".", "-"))
    add = add[~add.index.duplicated()].reindex(adj.columns)
    r, hist = adj.pct_change(fill_method=None), adj.notna().cumsum()
    me = adj.index.to_series().groupby(adj.index.to_period("M")).max()
    out = {}
    for prev, m0, m1 in zip(me.iloc[:-2], me.iloc[1:-1], me.iloc[2:]):
        nxt = adj.index[adj.index.get_loc(m0) + 1]
        if not (pd.Timestamp(start) <= nxt <= pd.Timestamp(end)):
            continue
        month = r[(r.index > prev) & (r.index <= m0)]
        mx = month.max().where(month.count() >= 15)
        ok = adj.loc[m0].notna() & (hist.loc[m0] >= 252) & ~(add > m0)
        s = mx[ok].dropna().rename("v").rename_axis("t").reset_index().sort_values(["v", "t"])
        names = list(s["t"].iloc[: -(-len(s) // 10)])
        out[m0] = (set(names), float((adj.loc[m1, names] / adj.loc[m0, names] - 1).mean()))
    return out


@pytest.fixture(scope="module")
def setup():
    ctx = DataContext.load()
    res = build_targets(ctx, max_spec())
    inv = {ex: form for form, ex in res.execution_days.items()}
    at_month_end = {inv[k]: w for k, w in res.targets.items()}          # execute at the formation close
    return ctx, res, at_month_end


def _monthly(values, month_ends):
    v = values.loc[month_ends]
    return (v.shift(-1) / v - 1).iloc[:-1]


def test_engine_matches_independent(setup):
    ctx, res, tg = setup
    ind = independent_max_decile1()
    forms = sorted(tg)
    assert forms == sorted(ind)
    for f in forms:
        assert set(tg[f].index) == ind[f][0]
    end = ctx.month_ends[ctx.month_ends.get_loc(forms[-1]) + 1]
    r = run_backtest(ctx, tg, 0, forms[0], end)
    mret = _monthly(r.values["net_value"], pd.DatetimeIndex(forms + [end]))
    exp = pd.Series({f: ind[f][1] for f in forms})
    assert np.abs(mret.to_numpy() - exp.to_numpy()).max() < 1e-10


def test_daily_retarget_differs_only_by_drift():
    # PARA's adjusted close in this snapshot is broken (e.g. -82% in Dec 2023, +645% on
    # 2025-06-16; see run.log data_warning); it alone pushes one month's drift gap to 1.6%.
    ctx = DataContext.load(exclude=("PARA",))
    res = build_targets(ctx, max_spec())
    inv = {ex: form for form, ex in res.execution_days.items()}
    tg = {inv[k]: w for k, w in res.targets.items()}
    forms = sorted(tg)
    end = ctx.month_ends[ctx.month_ends.get_loc(forms[-1]) + 1]
    days = ctx.calendar[(ctx.calendar >= forms[0]) & (ctx.calendar <= end)]
    keys = pd.DatetimeIndex(forms)
    daily = {d: tg[keys[keys.searchsorted(d, side="right") - 1]] for d in days}
    monthly = run_backtest(ctx, tg, 0, forms[0], end, record=False)
    retarget = run_backtest(ctx, daily, 0, forms[0], end, record=False)
    me = pd.DatetimeIndex(forms + [end])
    diff = (_monthly(monthly.values["net_value"], me) - _monthly(retarget.values["net_value"], me)).abs()
    assert diff.max() < 0.015 and diff.median() < 0.003
    assert diff.max() > 0                                    # they are not the same thing
    # and the daily re-targeted book is exactly constant-weight: net return = sum(w * r)
    for d in days[1:40]:
        w = daily[ctx.calendar[ctx.calendar.get_loc(d) - 1]]
        assert abs(retarget.returns.at[d, "net"] - (w * ctx.returns.loc[d, w.index]).sum()) < 1e-12
