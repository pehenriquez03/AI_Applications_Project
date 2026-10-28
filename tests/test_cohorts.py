"""Overlapping holding cohorts (holding_months = K > 1)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_engine.backtest.engine import run_backtest
from research_engine.contracts import ROOT, ExperimentSpec, RunFiles
from research_engine.data.loaders import DataContext
from research_engine.strategies.cohorts import net_cohort_targets
from research_engine.strategies.dev_run import eval_window, run_dev, summarize
from research_engine.strategies.portfolios import build_targets

K = 6
MOM_SPEC = ROOT / "runs" / "dev_momentum" / "experiment_spec.json"


def momentum_spec(**kw) -> ExperimentSpec:
    """runs/dev_momentum spec. adapt.py writes cum_return params as {months, skip}; the op takes
    skip_months (and no `name`), so the params are mapped in memory here (file untouched)."""
    s = ExperimentSpec.model_validate_json(MOM_SPEC.read_text())
    ops = [o.model_copy(update={"params": {("skip_months" if k == "skip" else k): v
                                           for k, v in o.params.items() if k != "name"}})
           for o in s.signal_ops]
    return s.model_copy(update={"signal_ops": ops, **kw})


@pytest.fixture(scope="module")
def ctx():
    return DataContext.load(exclude=("PARA",))


@pytest.fixture(scope="module")
def r6(ctx):
    return build_targets(ctx, momentum_spec())


@pytest.fixture(scope="module")
def r1(ctx):
    return build_targets(ctx, momentum_spec(holding_months=1))


# --------------------------------------------------------------------------- synthetic
def _synthetic(n_months=10, n=10, k=K, skip=(), drop=None):
    dates = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    ex = pd.Series(dates + pd.Timedelta(days=1), index=dates)
    universe = [f"T{i:02d}" for i in range(40)]
    # cohort m = names m..m+n-1, so consecutive cohorts overlap in n-1 names
    cohorts = {d: (None if i in skip else pd.Index(universe[i:i + n])) for i, d in enumerate(dates)}
    drop = drop or {}

    def elig(d):
        return pd.Index([t for t in universe if not (t in drop and d >= drop[t])])
    t, ev = net_cohort_targets(dates, ex, cohorts, k, elig)
    return dates, ex, t, ev


def test_synthetic_weights_sum_and_overlap():
    dates, ex, t, ev = _synthetic()
    assert not ev
    for i, d in enumerate(dates):
        w = t[ex[d]]
        assert abs(w.sum() - min(i + 1, K) / K) < 1e-12
    # (c) at month index 2 (3 live cohorts: names 0-9, 1-10, 2-11) T02..T09 sit in all three
    w = t[ex[dates[2]]]
    assert w["T05"] == pytest.approx(3 * (1 / K) * (1 / 10), abs=1e-15)
    assert w["T00"] == pytest.approx(1 / K / 10) and w["T11"] == pytest.approx(1 / K / 10)
    # steady state: cohort m-6 (names m-6..m+3) expired, T(m-6) no longer held
    w = t[ex[dates[8]]]
    assert "T02" not in w.index and w["T08"] == pytest.approx(6 / K / 10)


def test_synthetic_skip_carries_forward_and_drop_reweights():
    dates, ex, t, ev = _synthetic(skip={7})
    w7, w6 = t[ex[dates[7]]], t[ex[dates[6]]]
    assert abs(w7.sum() - 1) < 1e-12
    assert w7.equals(w6)          # the slot of month 7 keeps cohort 1 (formed 6 months earlier)
    assert [e["event"] for e in ev] == ["carry_forward"]
    dates, ex, t, ev = _synthetic(drop={"T05": dates[3]})
    w = t[ex[dates[3]]]
    assert "T05" not in w.index and abs(w.sum() - 4 / K) < 1e-12
    assert w["T00"] == pytest.approx(1 / K / 9)                 # cohort 0 re-equal-weighted over 9
    assert sorted(e["cohort"] for e in ev) == [d.date() for d in dates[:4]]
    assert all(e["event"] == "drop_ineligible" and e["ticker"] == "T05" for e in ev)


# --------------------------------------------------------------------------- real data
def test_k1_identical_to_existing_path(ctx, r1, r6):
    """(a) K=1 goes through the original path; netting a single cohort reproduces it exactly."""
    assert r1.holding_months == 1 and r1.cohort_log == []
    tickers = ctx.tickers.astype(str)
    elig = lambda d: tickers[ctx.eligible_mask(d, 252, 5.0)]  # noqa: E731
    for per_exec in [r1.targets] + list(r1.group_targets.values()):
        coh = {d: (per_exec[e].index if e in per_exec else None) for d, e in r1.execution_days.items()}
        t, ev = net_cohort_targets(r1.formation_dates, r1.execution_days, coh, 1, elig)
        assert not ev and t.keys() == per_exec.keys()
        assert all(t[k].equals(per_exec[k]) for k in t)
    for k, w in r1.targets.items():
        assert abs(w.sum() - 1) < 1e-12 and np.allclose(w, 1 / len(w))
    # signals_long (the per-month dossier table) does not depend on K
    pd.testing.assert_frame_equal(r1.signals_long, r6.signals_long)
    sl = r6.signals_long
    assert (sl["cohort"] == sl["date"]).all()
    n = sl.groupby("date")["selected"].transform("sum")
    assert np.allclose(sl.loc[sl["selected"], "weight"], 1 / n[sl["selected"]])


def test_k6_net_target_sums(r6):
    """(b) live-cohort fraction during start-up, 1 afterwards; the same for every group."""
    assert not r6.skipped_months
    days = [r6.execution_days[d] for d in r6.formation_dates]
    for tg in [r6.targets] + list(r6.group_targets.values()):
        assert list(tg) == days
        for i, e in enumerate(days):
            assert abs(tg[e].sum() - min(i + 1, K) / K) < 1e-12


def test_k6_name_weight_is_sum_of_cohort_weights(r1, r6):
    """(c) weight = (1/K) x sum over live cohorts holding the name of 1/n_cohort; a name in three
    equal-size cohorts gets 3 x (1/K) x (1/n)."""
    days = [r6.execution_days[d] for d in r6.formation_dates]
    dropped = {e["execution_day"] for e in r6.cohort_log if e["label"] == "strategy"}
    found3 = 0
    for i in range(K, len(days)):
        if any(days[j].date() in dropped for j in range(i - K + 1, i + 1)):
            continue
        live = [r1.targets[days[j]] for j in range(i - K + 1, i + 1)]
        exp = pd.concat([pd.Series(1 / (K * len(c)), index=c.index) for c in live]).groupby(level=0).sum()
        pd.testing.assert_series_equal(r6.targets[days[i]], exp, check_names=False, rtol=0, atol=1e-15)
        cnt = pd.concat([pd.Series(1, index=c.index) for c in live]).groupby(level=0).sum()
        for t in cnt.index[cnt == 3]:
            ns = {len(c) for c in live if t in c.index}
            if len(ns) == 1:
                assert r6.targets[days[i]][t] == pytest.approx(3 * (1 / K) * (1 / ns.pop()), abs=1e-15)
                found3 += 1
    assert found3 > 0


def test_k6_steady_state_turnover(ctx, r6, capsys):
    """(d) Turnover of each net target against the drifted previous book. The pure cohort
    replacement part is 0.5|w_t - w_{t-1}| (one cohort out, one in, net of overlap; <= 1/K);
    the rest is the drift correction added by resetting every cohort to 1/K and every name to
    1/n_cohort at each execution day."""
    start, end = eval_window(ctx, momentum_spec())
    bt = run_backtest(ctx, r6.targets, 0, start, end, record=False)
    days = [r6.execution_days[d] for d in r6.formation_dates]
    turn = bt.values["turnover"].reindex(days)
    steady = days[K:]                      # every slot filled and the previous target was full too
    total = turn.loc[steady]
    repl = pd.Series({e: 0.5 * r6.targets[e].sub(r6.targets[p], fill_value=0).abs().sum()
                      for p, e in zip(days[K - 1:-1], steady)})
    drift = total - repl
    stats = dict(total=total.mean(), replacement=repl.mean(), drift=drift.mean(), one_over_k=1 / K,
                 excess_vs_one_over_k=total.mean() - 1 / K, n_months=len(steady))
    with capsys.disabled():
        print("\nK=6 momentum steady-state monthly turnover: " + json.dumps({k: round(v, 4) for k, v in stats.items()}))
    assert (repl <= 1 / K + 1e-12).all()
    assert 0 < drift.mean() < 0.15
    assert abs(total.mean() - (repl.mean() + drift.mean())) < 1e-12
    assert 1 / K - 0.05 < total.mean() < 1 / K + 0.15


def _perturbed(ctx, after, seed=7):
    rng = np.random.default_rng(seed)
    mask = ctx.calendar > after
    out = {}
    for f in ("adj_close", "close", "high", "low"):
        p = getattr(ctx, f).copy()
        p.loc[mask] = p.loc[mask].to_numpy() * (1 + 0.05 * rng.standard_normal((int(mask.sum()), p.shape[1])))
        out[f] = p
    return ctx.replace_prices(**out)


def test_k6_no_look_ahead(ctx, r6):
    """(e) perturbing prices after D leaves every target dated on or before D unchanged."""
    D = pd.Timestamp("2019-03-15")
    r2 = build_targets(_perturbed(ctx, D), momentum_spec())
    before = [k for k in r6.targets if k <= D]
    assert len(before) > K
    for tg1, tg2 in [(r6.targets, r2.targets)] + [(r6.group_targets[g], r2.group_targets[g])
                                                  for g in r6.group_targets]:
        assert all(tg1[k].equals(tg2[k]) for k in tg1 if k <= D)
    assert any(not r6.targets[k].equals(r2.targets[k]) for k in r6.targets if k > D)


def test_k6_momentum_end_to_end(ctx, tmp_path, capsys):
    """(f) run_dev on the dev_momentum spec (written to a temp dir here; the run folder copy
    is produced by the same call outside the test suite)."""
    res = run_dev(momentum_spec(), tmp_path, ctx=ctx)
    r = pd.read_parquet(tmp_path / RunFiles.returns)
    assert {"strategy_net", "spy", "ew_universe"} | {f"group_{g}" for g in range(1, 11)} <= set(r.columns)
    assert res.holding_months == K and not r.isna().any().any()
    log = (tmp_path / RunFiles.run_log).read_text()
    assert "cohort event=drop_ineligible" in log
    with capsys.disabled():
        print("\n" + "\n".join(summarize(tmp_path)))
