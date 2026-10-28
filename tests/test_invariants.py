"""Backtest engine invariants."""
import numpy as np
import pandas as pd
import pytest

from research_engine.backtest.benchmarks import ew_universe_targets, spy_targets
from research_engine.backtest.engine import run_backtest
from research_engine.contracts import ExperimentSpec, SignalOp
from research_engine.data.loaders import DataContext
from research_engine.strategies.portfolios import build_targets

T = pd.Timestamp
START, END = "2015-01-01", "2018-12-31"


def max_spec(**kw):
    return ExperimentSpec(run_id="t", paper_sha256="t", strategy_slug="max", status="ready",
                          signal_ops=[SignalOp(op="rolling_max_daily_return")],
                          selection={"sort": "deciles", "n_groups": 10, "favored": "lowest"}, **kw)


@pytest.fixture(scope="module")
def ctx():
    return DataContext.load()


@pytest.fixture(scope="module")
def targets(ctx):
    return build_targets(ctx, max_spec()).targets


@pytest.fixture(scope="module")
def run10(ctx, targets):
    return run_backtest(ctx, targets, 10, START, END)


def test_value_identity(run10):
    h, v = run10.holdings, run10.values
    held = h.groupby("date")["value"].sum().reindex(v.index, fill_value=0.0)
    assert np.allclose(held + v["cash"], v["net_value"], rtol=0, atol=1e-12)
    assert np.allclose(h["shares"] * h["price"], h["value"], atol=1e-15)
    assert (v["n_holdings"] == h.groupby("date").size().reindex(v.index, fill_value=0)).all()
    r = v["net_value"] / v["net_value"].shift(1, fill_value=1.0) - 1
    assert np.allclose(r, run10.returns["net"], atol=1e-14)


def test_trades_only_on_execution_days(ctx, targets, run10):
    exec_days = {k for k in targets if T(START) <= k <= T(END)}
    tr = run10.trades
    assert set(tr.loc[tr["reason"] == "rebalance", "date"]) <= exec_days
    assert set(tr["reason"]) <= {"rebalance", "forced_sale"}
    # forced sale: make one held name ineligible mid-month
    k = sorted(exec_days)[3]
    victim = targets[k].index[0]
    cut = ctx.calendar[ctx.calendar.get_loc(k) + 5]
    fn = lambda d: ctx.tickers if d < cut else ctx.tickers.drop(victim)  # noqa: E731
    r = run_backtest(ctx, targets, 10, START, END, eligible_fn=fn)
    fs = r.trades[r.trades["reason"] == "forced_sale"]
    assert len(fs) == 1 and fs["date"].iloc[0] == cut and fs["ticker"].iloc[0] == victim
    other = r.trades[(r.trades["reason"] == "rebalance")]
    assert set(other["date"]) <= exec_days
    assert victim not in set(r.holdings.loc[r.holdings["date"] >= cut, "ticker"])


def test_cost_monotonicity(ctx, targets, run10):
    r0 = run_backtest(ctx, targets, 0, START, END, record=False)
    r25 = run_backtest(ctx, targets, 25, START, END, record=False)
    pd.testing.assert_series_equal(r0.returns["gross"], run10.returns["gross"], check_exact=True)
    pd.testing.assert_series_equal(r25.returns["gross"], run10.returns["gross"], check_exact=True)
    assert (r0.returns["gross"] == r0.returns["net"]).all()
    v0, v10, v25 = r0.values["net_value"], run10.values["net_value"], r25.values["net_value"]
    assert (v10 < v0).all() and (v25 < v10).all()
    assert abs(run10.trades["cost"].sum() - 10 / 1e4 * run10.trades["value"].abs().sum()) < 1e-15


def test_target_equal_to_drift_trades_nothing(ctx, targets):
    k0 = sorted(k for k in targets if k >= T(START))[0]
    base = run_backtest(ctx, {k0: targets[k0]}, 0, START, END)
    x = ctx.calendar[ctx.calendar.get_loc(k0) + 10]
    h = base.holdings[base.holdings["date"] == x].set_index("ticker")
    drifted = h["value"] / base.values.at[x, "net_value"]
    r = run_backtest(ctx, {k0: targets[k0], x: drifted}, 0, START, END)
    assert (r.trades["date"] == x).sum() == 0
    assert r.values.at[x, "turnover"] < 1e-12
    pd.testing.assert_frame_equal(r.holdings, base.holdings)


def test_new_weights_earn_nothing_before_execution(ctx, targets):
    keys = sorted(k for k in targets if k >= T(START))[:2]
    r = run_backtest(ctx, {k: targets[k] for k in keys}, 0, START, END)
    e = keys[1]
    prev = ctx.calendar[ctx.calendar.get_loc(e) - 1]
    h = r.holdings[r.holdings["date"] == prev].set_index("ticker")
    w_old = h["value"] / h["value"].sum()
    old_ret = (w_old * ctx.returns.loc[e, w_old.index]).sum()
    assert abs(r.returns.at[e, "net"] - old_ret) < 1e-12


def _perturbed(ctx, after, seed=11):
    rng = np.random.default_rng(seed)
    mask = ctx.calendar > after
    out = {}
    for f in ("adj_close", "close", "high", "low"):
        p = getattr(ctx, f).copy()
        p.loc[mask] = p.loc[mask].to_numpy() * (1 + 0.05 * rng.standard_normal((int(mask.sum()), p.shape[1])))
        out[f] = p
    return ctx.replace_prices(**out)


def test_look_ahead_mutation(ctx, run10):
    D = T("2017-06-14")
    c2 = _perturbed(ctx, D)
    r2 = run_backtest(c2, build_targets(c2, max_spec()).targets, 10, START, END)
    h1 = run10.holdings[run10.holdings["date"] <= D].reset_index(drop=True)
    h2 = r2.holdings[r2.holdings["date"] <= D].reset_index(drop=True)
    pd.testing.assert_frame_equal(h1, h2, check_exact=True)
    pd.testing.assert_frame_equal(run10.values.loc[:D], r2.values.loc[:D], check_exact=True)
    assert not run10.values.loc[D:].iloc[1:].equals(r2.values.loc[D:].iloc[1:])


def test_daily_constant_weight_matches_sum_wr(ctx):
    days = ctx.calendar[(ctx.calendar >= START) & (ctx.calendar <= END)]
    names = ctx.adj_close.loc[days].dropna(axis=1).columns[:30]
    w = pd.Series(np.random.default_rng(3).dirichlet(np.ones(30)), index=names)
    r = run_backtest(ctx, {d: w for d in days}, 0, START, END)
    exp = (ctx.returns.loc[days, names] * w).sum(axis=1)
    assert r.returns["net"].iloc[0] == 0.0
    assert np.abs(r.returns["net"].iloc[1:] - exp.iloc[1:]).max() < 1e-10


def test_spy_benchmark_is_buy_and_hold(ctx):
    r = run_backtest(ctx, spy_targets(ctx), 0, START, END)
    spy = ctx.spy_adj_close.loc[r.values.index]
    assert np.abs(r.values["net_value"] - spy / spy.iloc[0]).max() < 1e-10
    assert (r.trades["date"] == r.values.index[0]).all() and len(r.trades) == 1
    assert np.abs(r.returns["net"].iloc[1:] - ctx.spy_returns.loc[r.values.index[1:]]).max() < 1e-10


def test_ew_universe_targets(ctx):
    s = max_spec()
    tg = ew_universe_targets(ctx, s)
    k = sorted(tg)[0]
    assert abs(tg[k].sum() - 1) < 1e-12 and len(tg[k]) == len(ctx.eligible(T("2014-12-31")))
