"""Portfolio construction: groups, selection, execution days, skips, double sorts."""
import numpy as np
import pandas as pd
import pytest

from research_engine.contracts import ExperimentSpec, SignalOp
from research_engine.data.loaders import DataContext
from research_engine.strategies.portfolios import assign_groups, assign_percent_groups, build_targets


def spec(**kw):
    base = dict(run_id="t", paper_sha256="t", strategy_slug="max", status="ready",
                signal_ops=[SignalOp(op="rolling_max_daily_return")],
                selection={"sort": "deciles", "n_groups": 10, "favored": "lowest"})
    base.update(kw)
    return ExperimentSpec(**base)


@pytest.fixture(scope="module")
def ctx():
    return DataContext.load()


@pytest.fixture(scope="module")
def res(ctx):
    return build_targets(ctx, spec())


def test_assign_groups_equal_count():
    v = pd.Series(np.random.default_rng(0).standard_normal(305), index=[f"T{i:03d}" for i in range(305)])
    g = assign_groups(v, 10)
    sizes = g.value_counts()
    assert sizes.max() - sizes.min() <= 1 and sizes.sum() == 305
    for k in range(1, 10):
        assert v[g == k].max() < v[g == k + 1].min()
    p = assign_percent_groups(v, 0.3)
    assert (p == 3).sum() == (p == 1).sum() == round(0.3 * 305)
    assert v[p == 3].min() > v[p == 2].max()


def test_max_targets(ctx, res):
    assert len(res.targets) == 132 and not res.skipped_months
    sl = res.signals_long
    for form, ex in res.execution_days.items():
        assert ex == ctx.execution_day(form) and ex in ctx.first_days
        w = res.targets[ex]
        assert abs(w.sum() - 1) < 1e-12 and np.allclose(w, 1 / len(w))
        day = sl[sl["date"] == form]
        assert set(day.loc[day["selected"], "ticker"]) == set(w.index)
        assert day.loc[day["group"] == 1, "signal"].max() <= day.loc[day["group"] == 2, "signal"].min()
        assert len(day) <= res.eligible_counts[form]
    assert set(res.group_targets[1]) == set(res.targets)
    assert all(res.group_targets[1][k].equals(res.targets[k]) for k in res.targets)
    assert list(sl.columns[:7]) == ["date", "ticker", "signal", "group", "selected", "weight", "cohort"]
    assert (sl["cohort"] == sl["date"]).all()


def test_favored_highest_and_percent(ctx):
    r = build_targets(ctx, spec(selection={"sort": "deciles", "favored": "highest"}))
    assert r.favored_group == 10
    sl = r.signals_long
    d = sl["date"].iloc[0]
    day = sl[sl["date"] == d]
    assert set(day.loc[day["group"] == 10, "ticker"]) == set(r.targets[r.execution_days[d]].index)
    r = build_targets(ctx, spec(selection={"sort": "top_bottom_percent", "percent": 0.3, "favored": "highest"}))
    day = r.signals_long[r.signals_long["date"] == d]
    assert day["selected"].sum() == round(0.3 * len(day))


def test_skip_when_too_few_names(ctx):
    r = build_targets(ctx, spec(min_names=1000))
    assert r.targets == {} and len(r.skipped_months) == 132
    assert all(s["n_names"] < 1000 for s in r.skip_log)



def test_double_sort(ctx):
    ds = {"outer_op": "pret_12_2", "outer_sort": "quintiles", "outer_favored": "highest",
          "inner_op": "info_discreteness", "inner_sort": "terciles", "inner_favored": "lowest"}
    r = build_targets(ctx, spec(signal_ops=[SignalOp(op="pret_12_2"), SignalOp(op="info_discreteness")],
                                selection={"sort": "quintiles", "double_sort": ds}, min_names=15))
    sl = r.signals_long
    d = sl["date"].iloc[-1]
    day = sl[sl["date"] == d]
    top = day[day["outer_group"] == 5]
    assert top["outer_signal"].min() >= day.loc[day["outer_group"] == 4, "outer_signal"].max()
    sel = day[day["selected"]]
    assert set(sel["ticker"]) <= set(top["ticker"])
    assert sel["signal"].max() <= top.loc[top["group"] == 2, "signal"].min()
    assert len(sel) >= 15 and set(sel["ticker"]) == set(r.targets[r.execution_days[d]].index)
