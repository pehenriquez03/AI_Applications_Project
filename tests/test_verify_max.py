"""Independent verification of the MAX dev run (runs/dev_max).

Written by an independent verifier. Checks 1-3 and 5 rebuild signals, eligibility, deciles
and returns straight from the snapshot Parquet files with plain pandas (no engine code).
Checks 4, 7 read the run artifacts. Checks 6 and 8 rerun the pipeline into temp dirs.

Run: .venv/bin/python -m pytest tests/test_verify_max.py -s -q
Numbers are printed with the prefix "VERIFY".
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "snapshot_2026-10-28"
UNIV = ROOT / "data" / "universe" / "constituents.csv"
RUN = ROOT / "runs" / "dev_max"
EVAL_START, EVAL_END = pd.Timestamp("2015-01-01"), pd.Timestamp("2025-12-31")
COST = 10 / 1e4
MIN_HIST, MIN_DAYS, N_GROUPS = 252, 15, 10
PRICE_FLOOR = 5.0  # stated convention; the run's spec has price_floor = null (see check 2b)


def say(*a):
    print("VERIFY", *a)


# ----------------------------------------------------------------------------- raw data
@pytest.fixture(scope="module")
def raw():
    man = json.loads((SNAP / "manifest.json").read_text())
    uni = man["universe_tickers"]
    adj_all = pd.read_parquet(SNAP / "adj_close.parquet")
    adj_all.index = pd.DatetimeIndex(adj_all.index).as_unit("ns")
    close = pd.read_parquet(SNAP / "close.parquet")
    close.index = pd.DatetimeIndex(close.index).as_unit("ns")
    splits = pd.read_parquet(SNAP / "splits.parquet").fillna(0.0)
    splits.index = pd.DatetimeIndex(splits.index).as_unit("ns")
    adj = adj_all[uni]
    csv = pd.read_csv(UNIV)
    add = pd.Series(pd.to_datetime(csv["Date added"]).to_numpy(),
                    index=csv["Symbol"].astype(str).str.replace(".", "-", regex=False))
    add = add[~add.index.duplicated()].reindex(uni)
    rets = adj / adj.shift(1) - 1.0          # gaps stay NaN (no fill)
    rets[adj.shift(1).isna() | adj.isna()] = np.nan
    days = adj.index
    month_ends = pd.DatetimeIndex(days.to_series().groupby(days.to_period("M")).max().to_numpy())
    nxt = {d: days[days.get_loc(d) + 1] for d in month_ends[:-1]}
    forms = pd.DatetimeIndex([d for d in month_ends[:-1] if EVAL_START <= nxt[d] <= EVAL_END])
    return dict(adj=adj, spy=adj_all["SPY"], close=close[uni], splits=splits.reindex(columns=uni).fillna(0.0),
                add=add, rets=rets, days=days, forms=forms, nxt=nxt)


@pytest.fixture(scope="module")
def sig():
    s = pd.read_parquet(RUN / "signals.parquet")
    s["ticker"] = s["ticker"].astype(str)
    return s


def my_max(raw, d):
    """Max daily return over the trading days of d's calendar month up to d; NaN if < 15 obs."""
    r = raw["rets"]
    p = d.to_period("M")
    block = r[(r.index.to_period("M") == p) & (r.index <= d)]
    return block.max().where(block.count() >= MIN_DAYS)


def my_eligible(raw, d, price_floor=None, raw_close=None):
    adj = raw["adj"]
    hist = adj.loc[:d].notna().sum()
    ok = adj.loc[d].notna() & (hist >= MIN_HIST) & ~(raw["add"] > d)
    if price_floor is not None:
        px = (raw_close if raw_close is not None else raw["close"]).loc[d]
        ok &= px >= price_floor
    return ok[ok].index


def my_decile1(raw, d, price_floor=None):
    v = my_max(raw, d).reindex(my_eligible(raw, d, price_floor)).dropna()
    df = pd.DataFrame({"t": v.index.astype(str), "v": v.to_numpy()}).sort_values(["v", "t"])
    k = math.ceil(len(df) / N_GROUPS)  # equal-count deciles: lowest ceil(n/10) names
    return set(df["t"].iloc[:k]), len(df)


# ----------------------------------------------------------------------------- 1 signal
@pytest.mark.parametrize("d", ["2020-08-31", "2021-07-30", "2024-06-28"])
def test_1_signal_by_hand(raw, sig, d):
    """AAPL split 2020-08-31 (on the month-end itself); NVDA splits 2021-07-20 and 2024-06-10."""
    d = pd.Timestamp(d)
    adj = raw["adj"]
    month = adj[(adj.index.to_period("M") == d.to_period("M")) & (adj.index <= d)]
    prev_close = adj[adj.index < month.index[0]].iloc[-1]
    s = sig[sig["date"] == d].set_index("ticker")["signal"]
    for t in ["AAPL", "MSFT", "XOM", "JPM", "NVDA"]:
        px = pd.concat([prev_close[[t]].to_frame().T, month[[t]]])[t].to_numpy()
        hand = max(px[i] / px[i - 1] - 1 for i in range(1, len(px)))
        diff = abs(hand - s[t])
        say(f"1 {d.date()} {t} hand={hand:.10f} engine={s[t]:.10f} diff={diff:.2e}")
        assert diff < 1e-10


# ----------------------------------------------------------------------------- 2 construction
@pytest.mark.parametrize("d", ["2015-06-30", "2020-03-31", "2025-06-30"])
def test_2_construction(raw, sig, d):
    d = pd.Timestamp(d)
    mine, n = my_decile1(raw, d)
    s = sig[sig["date"] == d]
    eng = set(s.loc[s["selected"], "ticker"])
    elig_mine = set(my_eligible(raw, d))
    elig_eng = set(s["ticker"])  # signals rows = eligible names with a signal
    say(f"2 {d.date()} n_ranked mine={n} engine={len(s)} decile1 mine={len(mine)} engine={len(eng)} "
        f"only_mine={sorted(mine - eng)} only_engine={sorted(eng - mine)}")
    assert elig_eng <= elig_mine
    assert mine == eng


def test_2_all_months_and_groups(raw, sig):
    """All 132 formation months: decile-1 set equality, plus full group labels."""
    bad = []
    for d in raw["forms"]:
        mine, _ = my_decile1(raw, d)
        s = sig[sig["date"] == d]
        if mine != set(s.loc[s["selected"], "ticker"]):
            bad.append(d.date())
    say(f"2 all-months: {len(raw['forms'])} formation months, mismatching decile-1 sets: {bad}")
    assert not bad


def test_2b_price_floor_effect(raw):
    """Convention says close >= $5, the spec has price_floor=null. Report what a floor would do,
    on Yahoo's back-split-adjusted close and on a reconstructed raw close."""
    sp = raw["splits"].where(raw["splits"] > 0, 1.0)
    later = sp.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)  # product of splits after each date
    raw_close = raw["close"] * later
    n_adj = n_raw = 0
    changed_adj, changed_raw = [], []
    for d in raw["forms"]:
        base = set(my_eligible(raw, d))
        a = base - set(my_eligible(raw, d, PRICE_FLOOR))
        r = base - set(my_eligible(raw, d, PRICE_FLOOR, raw_close))
        n_adj += len(a); n_raw += len(r)
        if a:
            changed_adj.append((str(d.date()), sorted(a)))
        if r:
            changed_raw.append((str(d.date()), sorted(r)))
    say(f"2b $5 floor on Yahoo split-adjusted close would drop {n_adj} name-months in "
        f"{len(changed_adj)} months; first: {changed_adj[:3]}")
    say(f"2b $5 floor on reconstructed raw close would drop {n_raw} name-months in "
        f"{len(changed_raw)} months; first: {changed_raw[:3]}")
    d1 = []
    for d in raw["forms"]:
        a, _ = my_decile1(raw, d)
        b, _ = my_decile1(raw, d, PRICE_FLOOR)
        if a != b:
            d1.append(str(d.date()))
    say(f"2b decile-1 sets that change with a $5 floor on Yahoo close: {len(d1)} months {d1[:5]}")


# ----------------------------------------------------------------------------- 3 returns
def period_bounds(raw):
    days = raw["days"]
    ex = [raw["nxt"][d] for d in raw["forms"]]
    last = days[days <= EVAL_END][-1]
    return ex, ex[1:] + [last]


def my_period_returns(raw, members_fn):
    adj = raw["adj"].ffill()            # held name with no price keeps its last price (forced sale)
    adj_exact = raw["adj"]
    ex, ends = period_bounds(raw)
    out = {}
    for d, e0, e1 in zip(raw["forms"], ex, ends):
        names = [t for t in members_fn(d) if pd.notna(adj_exact.at[e0, t])]
        out[e1] = float((adj.loc[e1, names] / adj.loc[e0, names] - 1).mean())
    return pd.Series(out)


def compound(daily: pd.Series, raw):
    ex, ends = period_bounds(raw)
    return pd.Series({e1: float((1 + daily[(daily.index > e0) & (daily.index <= e1)]).prod() - 1)
                      for e0, e1 in zip(ex, ends)})


def test_3_independent_returns(raw):
    mine = my_period_returns(raw, lambda d: my_decile1(raw, d)[0])
    eng = compound(pd.read_parquet(RUN / "returns_cost_0.parquet")["strategy_gross"], raw)
    diff = (mine - eng).abs()
    say(f"3 decile-1 periods={len(diff)} max|diff|={diff.max():.2e} median|diff|={diff.median():.2e} "
        f"worst={diff.idxmax().date()}")
    assert diff.max() < 1e-8


# ----------------------------------------------------------------------------- 4 costs
def test_4_costs():
    tr = pd.read_parquet(RUN / "trades.parquet")
    pv = pd.read_parquet(RUN / "portfolio_value.parquet")
    r = pd.read_parquet(RUN / "returns.parquet")
    by = tr.groupby("date").agg(cost=("cost", "sum"), absval=("value", lambda x: x.abs().sum()),
                                netval=("value", "sum"))
    e1 = (by["cost"] - COST * by["absval"]).abs().max()
    row = (tr["cost"] - COST * tr["value"].abs()).abs().max()
    dcash = pv["cash"].diff().reindex(by.index)
    dcash.iloc[0] = pv["cash"].iloc[0] - 1.0
    e2 = (dcash - (-by["netval"] - by["cost"])).abs().max()
    vprev = pv["net_value"].shift(1).fillna(1.0).reindex(by.index)
    drag = by["cost"] / vprev
    gap = (r["strategy_net"] - r["strategy_gross"]).reindex(by.index)
    resid = (gap + drag).abs()
    non_exec = (r["strategy_net"] - r["strategy_gross"]).drop(by.index).abs()
    hv = pd.read_parquet(RUN / "holdings.parquet").groupby("date")["value"].sum()
    e3 = (hv + pv["cash"].reindex(hv.index) - pv["net_value"].reindex(hv.index)).abs().max()
    say(f"4 exec days={len(by)} max|cost-10bp*|trade$||: per-day {e1:.2e}, per-row {row:.2e}; "
        f"cash identity {e2:.2e}; holdings+cash=net_value {e3:.2e}")
    say(f"4 net-gross on exec days vs -cost/V_prev: max resid {resid.max():.2e} median {resid.median():.2e}; "
        f"mean drag {drag.mean():.5f}; non-exec |net-gross| max {non_exec.max():.2e}; "
        f"cash after rebalance (min) {pv['cash'].min():.5f}")
    assert e1 < 1e-12 and row < 1e-12 and e2 < 1e-10 and e3 < 1e-10
    assert resid.max() < 5e-4  # only the net/gross book divergence from cash-financed costs remains


# ----------------------------------------------------------------------------- 5 benchmarks
def test_5_benchmarks(raw):
    from research_engine.backtest.benchmarks import ew_universe_targets, spy_targets
    from research_engine.backtest.engine import run_backtest
    from research_engine.data.loaders import DataContext
    from research_engine.strategies.dev_run import eval_window, max_spec
    ctx, spec = DataContext.load(), max_spec()
    start, end = eval_window(ctx, spec)
    ew0 = run_backtest(ctx, ew_universe_targets(ctx, spec), 0, start, end, record=False).returns["net"]
    mine = my_period_returns(raw, lambda d: my_eligible(raw, d))
    diff = (mine - compound(ew0, raw)).abs()
    say(f"5 ew_universe @0bp vs mine: max|diff|={diff.max():.2e} median={diff.median():.2e}")
    r = pd.read_parquet(RUN / "returns.parquet")
    ew10 = compound(r["ew_universe"], raw)
    say(f"5 ew_universe @10bp (returns.parquet) mean monthly shortfall vs mine {(mine - ew10).mean():.2e}")
    spy_bh = raw["spy"].pct_change().loc[start:end]
    spy0 = run_backtest(ctx, spy_targets(ctx), 0, start, end, record=False).returns["net"]
    d0 = (spy0.iloc[1:] - spy_bh.iloc[1:]).abs().max()
    d10 = (r["spy"].iloc[1:] - spy_bh.iloc[1:]).abs().max()
    tv_eng = (1 + r["spy"]).prod()
    tv_true = (1 - COST) * raw["spy"].loc[end] / raw["spy"].loc[start]
    say(f"5 spy @0bp vs SPY adj_close buy-and-hold: max|diff|={d0:.2e}, day-1 return {spy0.iloc[0]:.2e}")
    say(f"5 spy @10bp (returns.parquet): day-1 {r['spy'].iloc[0]:.5f}, later max|diff|={d10:.2e}; "
        f"terminal value engine {tv_eng:.6f} vs (1-10bp)*P_T/P_0 {tv_true:.6f}")
    assert diff.max() < 1e-8
    assert d0 < 1e-12 and abs(spy0.iloc[0]) < 1e-12


# ----------------------------------------------------------------------------- 6 look-ahead
def test_6_lookahead(tmp_path):
    from research_engine.data.loaders import DataContext
    from research_engine.strategies.dev_run import max_spec, run_dev
    cut = pd.Timestamp("2020-06-30")
    ctx = DataContext.load(fresh=True)
    rng = np.random.default_rng(7)
    adj = ctx.adj_close.copy()
    after = adj.index > cut
    adj.loc[after] = adj.loc[after].to_numpy() * rng.uniform(0.5, 1.5, size=adj.loc[after].shape)
    noisy = ctx.replace_prices(adj_close=adj)
    run_dev(max_spec(), tmp_path, ctx=noisy)
    for name in ("signals", "holdings", "trades"):
        a = pd.read_parquet(RUN / f"{name}.parquet")
        b = pd.read_parquet(tmp_path / f"{name}.parquet")
        a, b = a[a["date"] <= cut].reset_index(drop=True), b[b["date"] <= cut].reset_index(drop=True)
        later_differs = not pd.read_parquet(RUN / f"{name}.parquet").reset_index(drop=True).equals(
            pd.read_parquet(tmp_path / f"{name}.parquet").reset_index(drop=True))
        say(f"6 {name}: rows<=cut {len(a)} vs {len(b)}, identical={a.equals(b)}, full file differs={later_differs}")
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    ra = pd.read_parquet(RUN / "returns.parquet").loc[:cut]
    rb = pd.read_parquet(tmp_path / "returns.parquet").loc[:cut]
    say(f"6 returns<=cut identical={ra.equals(rb)}")
    pd.testing.assert_frame_equal(ra, rb, check_exact=True)


# ----------------------------------------------------------------------------- 7 sanity
def test_7_sanity():
    r = pd.read_parquet(RUN / "returns.parquet")
    g = [f"group_{i}" for i in range(1, 11)]
    vol = r[g].std() * np.sqrt(252)
    beta = r[g].apply(lambda x: np.cov(x, r["spy"])[0, 1] / r["spy"].var())
    turn = pd.read_parquet(RUN / "portfolio_value.parquet")["turnover"].dropna()
    rho = pd.Series(range(10)).corr(pd.Series(vol.to_numpy()), method="spearman")
    say("7 ann vol by decile: " + " ".join(f"{x:.3f}" for x in vol))
    say("7 beta by decile:    " + " ".join(f"{x:.2f}" for x in beta))
    say(f"7 vol spearman vs decile {rho:.3f}; turnover mean all {turn.mean():.3f}, "
        f"excl. initial buy {turn.iloc[1:].mean():.3f} (one-sided, n={len(turn)})")
    say(f"7 group_1 == strategy_net: {np.allclose(r['group_1'], r['strategy_net'], atol=1e-15, rtol=0)}")
    assert vol["group_10"] > vol["group_1"] and rho > 0.9
    assert beta["group_1"] < beta["group_10"]
    assert 0.6 <= turn.iloc[1:].mean() <= 0.9


# ----------------------------------------------------------------------------- 8 PARA
def test_8_para(tmp_path):
    from research_engine.data.loaders import DataContext
    from research_engine.strategies.dev_run import max_spec, run_dev

    def cagr(x):
        return (1 + x).prod() ** (252 / len(x)) - 1
    base = pd.read_parquet(RUN / "returns.parquet")
    ctx = DataContext.load(exclude=("PARA",), fresh=True)
    assert "PARA" not in ctx.tickers
    run_dev(max_spec(), tmp_path, ctx=ctx)
    ex = pd.read_parquet(tmp_path / "returns.parquet")
    out = {c: (cagr(base[c]), cagr(ex[c])) for c in ("strategy_net", "ew_universe", "group_10")}
    for c, (a, b) in out.items():
        say(f"8 {c}: with PARA {a:.4%} without {b:.4%} delta {100 * (b - a):+.3f} pp")
    s = pd.read_parquet(RUN / "signals.parquet")
    p = s[(s["ticker"] == "PARA") & (s["date"] >= "2025-05-01")][["date", "signal", "group", "selected"]]
    say("8 PARA 2025 signals:\n" + p.to_string(index=False))
    a, b = out["strategy_net"]
    assert abs(b - a) * 100 <= 0.1
