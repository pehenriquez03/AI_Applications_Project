"""Strategy scorecard: performance, relative performance, factor alpha, turnover,
costs, concentration, capacity, and robustness for one run folder.

`score_strategy(run_dir)` reads the run's parquet/json files plus the snapshot's
factor and volume files, writes `scorecard_strategy.json`, and returns the dict.
Every number is a plain float rounded to 6 decimals (NaN becomes null).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research_engine.contracts import (
    COST_BPS_BASE, COST_BPS_GRID, EVAL_END, EVAL_START, FUND, SNAPSHOT_DIR, SUBPERIOD_SPLIT, RunFiles,
)
from research_engine.evaluation.stats import (
    alpha_regression, annualize, monthly_from_daily, newey_west_tstat, spearman_monotonicity,
)

FUND_AUM = FUND["starting_aum"]
FACTOR_COLS = ["mkt_rf", "smb", "hml", "mom"]
Z95 = 1.96

DEFINITIONS = {
    "window": "All statistics use daily returns over the run's evaluation window (experiment_spec eval_start to eval_end, inclusive).",
    "cagr": "end_wealth ** (252 / n_days) - 1, compounding daily returns.",
    "vol": "standard deviation of daily returns (ddof=1) times sqrt(252).",
    "sharpe": "mean daily return in excess of the daily risk-free rate (Fama-French RF; zero if unavailable) times 252, divided by vol.",
    "max_drawdown": "most negative peak-to-trough change of the wealth curve, which starts at 1; reported as a negative number.",
    "end_wealth": "growth of 1 over the window.",
    "relative": "strategy_net minus the benchmark, computed on calendar-month compounded returns; "
                "t-statistic and se are Newey-West (Bartlett) with lags = holding_months; "
                "ci95 = mean +/- 1.96 se; annualized_diff = 12 * mean monthly difference; cagr_diff = cagr difference.",
    "beta_vs_spy": "OLS slope of daily strategy_net on daily spy.",
    "alpha": "four-factor regression of daily (strategy_net - rf) on mkt_rf, smb, hml, mom; Newey-West lags = 21 * holding_months; "
             "alpha_annual = daily intercept * 252.",
    "spanning": "four-factor regression of daily (strategy_net - ew_universe) on mkt_rf, smb, hml, mom; "
                "same lags; a positive alpha means the strategy adds return beyond the equal-weight universe and factor exposures.",
    "turnover": "portfolio_value.turnover on execution days (one-way, as written by the engine); the first execution day (initial build from cash) is excluded.",
    "costs": "total_cost = sum of trades.cost over the window; pct_of_avg_value = total_cost / mean net_value; annualized divides by years = n_days / 252.",
    "holdings": "mean_n_holdings = mean of portfolio_value.n_holdings; max_weight = largest single weight in holdings.parquet; "
                "mean_hhi = mean over dates of the sum of squared weights.",
    "capacity": "per trade, participation = |trade value| * fund_aum / (20-day average dollar volume of the ticker, close * volume, "
                "ending the prior trading day); fund_aum scales the backtest's $1 book to the stated fund size. "
                "per_trade gives mean, median, 95th/99th percentile and max over trades (max_trade names the worst one); per_day_sum sums participations across the trades of one execution day. "
                "Trades without 20 days of volume history are counted in n_missing_adv and excluded.",
    "cost_grid": "strategy_net re-run at each cost in bps (0, 10, 25); alpha_t is the four-factor alpha t-statistic.",
    "subperiods": "first = window start to the day before 2020-01-01; second = 2020-01-01 to the window end; when that split leaves a side under 12 months the window is split at its midpoint; "
                  "rel_ew_t is the Newey-West t of strategy_net minus ew_universe (monthly, lags = holding_months).",
    "groups": "group_k = k-th ranked group (1 = lowest signal), net of costs; mean_monthly = mean of monthly compounded returns; "
              "cagr_daily from daily compounding; rho = Spearman correlation of group index with mean monthly return; "
              "expected_rho_sign is -1 when favored='lowest' and expected_sign='favored_group_higher' (or favored='highest' with "
              "'favored_group_lower'), +1 for the other two cases; rho_in_expected_direction = rho * expected_rho_sign > 0 "
              "(null when expected_sign is unclear or rho undefined). favored_minus_disfavored is the mean monthly difference "
              "between the favored end group and the opposite end group with its Newey-West t.",
    "universe_sensitivity": "not computed here; the pipeline may fill it.",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _clean(o):
    """Recursively convert to JSON-safe python: floats rounded to 6, NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (pd.Timestamp,)):
        return o.strftime("%Y-%m-%d")
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    if isinstance(o, (pd.Series, pd.DataFrame)):
        return None
    return o


def _read_wide(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _window(df, start=None, end=None):
    start, end = start or EVAL_START, end or EVAL_END
    return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


def load_factors(snapshot_dir: Path) -> pd.DataFrame | None:
    """Daily decimal factors with columns mkt_rf, smb, hml, mom, rf; None if files are absent."""
    ff, mom = Path(snapshot_dir) / "ff3_daily.parquet", Path(snapshot_dir) / "mom_daily.parquet"
    if not ff.exists() or not mom.exists():
        return None
    f = _read_wide(ff).rename(columns={"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml", "RF": "rf"})
    m = _read_wide(mom)
    m = m.rename(columns={m.columns[0]: "mom"})
    return f.join(m, how="inner")[FACTOR_COLS + ["rf"]]


def _load_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def _perf(r: pd.Series, rf: pd.Series | None) -> dict:
    a = annualize(r, rf)
    return {k: a[k] for k in ("cagr", "vol", "sharpe", "max_drawdown", "end_wealth", "n_days")}


def _relative(strat_d: pd.Series, bench_d: pd.Series, lags: int) -> dict:
    ms, mb = monthly_from_daily(strat_d), monthly_from_daily(bench_d)
    diff = (ms - mb).dropna()
    mean, t, se = newey_west_tstat(diff, lags)
    cagr_diff = annualize(strat_d)["cagr"] - annualize(bench_d)["cagr"]
    return dict(n_months=len(diff), mean_monthly_diff=mean, nw_t=t, nw_se=se, nw_lags=lags,
                ci95_low=mean - Z95 * se, ci95_high=mean + Z95 * se,
                annualized_diff=12 * mean, cagr_diff=cagr_diff)


def _alpha_block(y: pd.Series, factors: pd.DataFrame | None, lags: int) -> dict | None:
    if factors is None:
        return None
    out = alpha_regression(y, factors[FACTOR_COLS], lags)
    out["nw_lags"] = lags
    return out


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def _turnover(pv: pd.DataFrame | None) -> dict | None:
    if pv is None or "turnover" not in pv:
        return None
    t = pv["turnover"].dropna()
    t = t[t > 0]
    t = t.groupby(t.index.to_period("M")).sum()
    if len(t) > 1:
        t = t.iloc[1:]
    if t.empty:
        return dict(mean_monthly=float("nan"), max_monthly=float("nan"), n_months=0)
    return dict(mean_monthly=float(t.mean()), max_monthly=float(t.max()), n_months=int(len(t)))


def _costs(trades: pd.DataFrame | None, pv: pd.DataFrame | None, n_days: int) -> dict | None:
    if trades is None or "cost" not in trades:
        return None
    total = float(trades["cost"].sum())
    avg_val = float(pv["net_value"].mean()) if pv is not None and "net_value" in pv else float("nan")
    years = n_days / 252.0
    pct = total / avg_val if avg_val and not math.isnan(avg_val) else float("nan")
    return dict(total_cost=total, avg_net_value=avg_val, pct_of_avg_value=pct,
                annualized_pct_of_avg_value=pct / years if years > 0 else float("nan"))


def _holdings(holdings: pd.DataFrame | None, pv: pd.DataFrame | None) -> dict | None:
    out = {}
    if pv is not None and "n_holdings" in pv:
        out["mean_n_holdings"] = float(pv["n_holdings"].mean())
        out["min_n_holdings"] = float(pv["n_holdings"].min())
    if holdings is not None and "weight" in holdings:
        out["max_weight"] = float(holdings["weight"].max())
        out["mean_hhi"] = float((holdings["weight"] ** 2).groupby(holdings["date"]).sum().mean())
    return out or None


def _capacity(trades: pd.DataFrame | None, snapshot_dir: Path, fund_aum: float) -> dict | None:
    vol_p, close_p = Path(snapshot_dir) / "volume.parquet", Path(snapshot_dir) / "close.parquet"
    if trades is None or trades.empty or not vol_p.exists() or not close_p.exists():
        return None
    tr = trades[["date", "ticker", "value"]].copy()
    tr["date"] = pd.to_datetime(tr["date"])
    tr = tr[tr["value"].abs() > 0]
    if tr.empty:
        return None
    tickers = sorted(tr["ticker"].unique())
    vol, close = _read_wide(vol_p), _read_wide(close_p)
    tickers = [t for t in tickers if t in vol.columns and t in close.columns]
    adv = (close[tickers] * vol[tickers]).rolling(20, min_periods=20).mean().shift(1)
    adv_long = adv.stack(future_stack=True)
    adv_long.index.names = ["date", "ticker"]
    key = pd.MultiIndex.from_arrays([tr["date"], tr["ticker"]], names=["date", "ticker"])
    tr["adv"] = adv_long.reindex(key).to_numpy()
    ok = tr["adv"].notna() & (tr["adv"] > 0)
    n_missing = int((~ok).sum())
    tr = tr[ok].copy()
    if tr.empty:
        return dict(fund_aum=fund_aum, n_trades=0, n_missing_adv=n_missing)
    tr["part"] = tr["value"].abs() * fund_aum / tr["adv"]
    per_day = tr.groupby("date")["part"].sum()

    def _q(s):
        return dict(mean=float(s.mean()), median=float(s.median()), p95=float(s.quantile(0.95)),
                    p99=float(s.quantile(0.99)), max=float(s.max()))

    worst = tr.loc[tr["part"].idxmax()]

    return dict(fund_aum=fund_aum, n_trades=int(len(tr)), n_missing_adv=n_missing,
                per_trade=_q(tr["part"]), per_day_sum=_q(per_day),
                max_trade=dict(date=worst["date"], ticker=worst["ticker"], adv_usd=float(worst["adv"]),
                               participation=float(worst["part"])),
                frac_trades_over_10pct_adv=float((tr["part"] > 0.10).mean()))


def _expected_rho_sign(selection: dict, expected_sign: str) -> int | None:
    favored = (selection or {}).get("favored")
    if favored not in ("lowest", "highest") or expected_sign not in ("favored_group_higher", "favored_group_lower"):
        return None
    favored_high_idx = favored == "highest"
    favored_higher_ret = expected_sign == "favored_group_higher"
    return 1 if favored_high_idx == favored_higher_ret else -1


def _group_table(rets: pd.DataFrame, selection: dict, expected_sign: str, lags: int) -> dict | None:
    selection = dict(selection or {})
    if selection.get("double_sort"):
        selection["favored"] = selection["double_sort"].get("inner_favored", "lowest")
    cols = sorted([c for c in rets.columns if c.startswith("group_")], key=lambda c: int(c.split("_")[1]))
    if not cols:
        return None
    rows, means = [], []
    monthly = {}
    for c in cols:
        m = monthly_from_daily(rets[c])
        monthly[c] = m
        means.append(float(m.mean()))
        rows.append(dict(group=int(c.split("_")[1]), mean_monthly=float(m.mean()),
                         cagr_daily=annualize(rets[c])["cagr"], annualized_mean_x12=12 * float(m.mean())))
    rho, p = spearman_monotonicity(means)
    sign = _expected_rho_sign(selection, expected_sign)
    in_dir = None if (sign is None or math.isnan(rho)) else bool(rho * sign > 0)
    out = dict(rows=rows, rho=rho, rho_p=p, n_groups=len(cols), favored=(selection or {}).get("favored"),
               expected_sign=expected_sign, expected_rho_sign=sign, rho_in_expected_direction=in_dir)
    fav = (selection or {}).get("favored")
    out["favored_group"] = int((cols[0] if fav == "lowest" else cols[-1]).split("_")[1]) if fav in ("lowest", "highest") else None
    if fav in ("lowest", "highest"):
        a, b = (cols[0], cols[-1]) if fav == "lowest" else (cols[-1], cols[0])
        diff = (monthly[a] - monthly[b]).dropna()
        mean, t, se = newey_west_tstat(diff, lags)
        out["favored_minus_disfavored"] = dict(favored_group=int(a.split("_")[1]), disfavored_group=int(b.split("_")[1]),
                                               mean_monthly_diff=mean, nw_t=t, nw_lags=lags)
    return out


def _cost_grid(run_dir: Path, base: pd.DataFrame, factors, lags_alpha: int, start=None, end=None) -> dict:
    out = {}
    for bps in COST_BPS_GRID:
        if bps == COST_BPS_BASE:
            s = base["strategy_net"]
        else:
            p = run_dir / f"returns_cost_{bps}.parquet"
            if not p.exists():
                out[str(bps)] = None
                continue
            df = _window(_read_wide(p), start, end)
            s = df["strategy_net"] if "strategy_net" in df else df.iloc[:, 0]
        rf = factors["rf"] if factors is not None else None
        a = annualize(s, rf)
        al = _alpha_block((s - factors["rf"].reindex(s.index)) if factors is not None else s, factors, lags_alpha)
        out[str(bps)] = dict(cagr=a["cagr"], sharpe=a["sharpe"], alpha_t=al["alpha_t"] if al else None,
                             alpha_annual=al["alpha_annual"] if al else None)
    return out


def _subperiods(r: pd.DataFrame, rf, lags: int) -> dict:
    split = pd.Timestamp(SUBPERIOD_SPLIT)
    if len(r) and not (r.index[0] <= split - pd.DateOffset(months=12) and split + pd.DateOffset(months=12) <= r.index[-1]):
        split = r.index[len(r) // 2]       # the fixed split leaves a side under a year: split the window at its midpoint
    out = {}
    for name, part in (("first", r[r.index < split]), ("second", r[r.index >= split])):
        if len(part) < 2:
            out[name] = None
            continue
        a = annualize(part["strategy_net"], rf)
        rel = _relative(part["strategy_net"], part["ew_universe"], lags) if "ew_universe" in part else None
        out[name] = dict(start=part.index[0], end=part.index[-1], cagr=a["cagr"], sharpe=a["sharpe"],
                         rel_ew_mean_monthly=rel["mean_monthly_diff"] if rel else None,
                         rel_ew_t=rel["nw_t"] if rel else None)
    return out


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def score_strategy(run_dir, snapshot_dir: Path | None = None, fund_aum: float | None = None,
                   write: bool = True) -> dict:
    run_dir = Path(run_dir)
    snapshot_dir = Path(snapshot_dir) if snapshot_dir else SNAPSHOT_DIR
    fund_aum = float(fund_aum if fund_aum is not None else FUND_AUM)

    exp = _load_json(run_dir / RunFiles.experiment_spec, {}) or {}
    start = str(exp.get("eval_start") or EVAL_START)[:10]
    end = str(exp.get("eval_end") or EVAL_END)[:10]
    rets = _window(_read_wide(run_dir / RunFiles.returns), start, end)
    holding_months = int(exp.get("holding_months", 1) or 1)
    selection = exp.get("selection") or {}
    expected_sign = exp.get("expected_sign", "unclear")
    lags_rel, lags_alpha = holding_months, 21 * holding_months

    factors = load_factors(snapshot_dir)
    if factors is not None:
        factors = _window(factors, start, end)
    rf = factors["rf"] if factors is not None else None

    pv = trades = holdings = None
    if (run_dir / RunFiles.portfolio_value).exists():
        pv = _window(_read_wide(run_dir / RunFiles.portfolio_value), start, end)
    if (run_dir / RunFiles.trades).exists():
        trades = pd.read_parquet(run_dir / RunFiles.trades)
        trades["date"] = pd.to_datetime(trades["date"])
        trades = trades[(trades["date"] >= start) & (trades["date"] <= end)]
    if (run_dir / RunFiles.holdings).exists():
        holdings = pd.read_parquet(run_dir / RunFiles.holdings)
        holdings["date"] = pd.to_datetime(holdings["date"])
        holdings = holdings[(holdings["date"] >= start) & (holdings["date"] <= end)]

    net = rets["strategy_net"]
    performance = {c: _perf(rets[c], rf) for c in ("strategy_gross", "strategy_net", "spy", "ew_universe") if c in rets}

    relative = {}
    for b in ("spy", "ew_universe"):
        if b in rets:
            relative[f"vs_{b}"] = _relative(net, rets[b], lags_rel)

    beta_vs_spy = None
    if "spy" in rets:
        d = pd.concat([net, rets["spy"]], axis=1, join="inner").dropna()
        v = d.iloc[:, 1].var()
        beta_vs_spy = float(d.iloc[:, 0].cov(d.iloc[:, 1]) / v) if v > 0 else float("nan")

    alpha = spanning = None
    if factors is not None:
        alpha = _alpha_block(net - rf.reindex(net.index), factors, lags_alpha)
        if "ew_universe" in rets:
            spanning = _alpha_block(net - rets["ew_universe"], factors, lags_alpha)

    sc = dict(
        meta=dict(run_dir=run_dir.name, eval_start=start, eval_end=end, n_days=len(rets),
                  first_date=rets.index[0], last_date=rets.index[-1], holding_months=holding_months,
                  expected_sign=expected_sign, selection=selection, factors_available=factors is not None,
                  fund_aum=fund_aum, cost_bps_base=COST_BPS_BASE),
        performance=performance,
        relative=relative,
        beta_vs_spy=beta_vs_spy,
        alpha=alpha,
        spanning=spanning,
        turnover=_turnover(pv),
        costs=_costs(trades, pv, len(rets)),
        holdings=_holdings(holdings, pv),
        capacity=_capacity(trades, snapshot_dir, fund_aum),
        robustness=dict(
            cost_grid=_cost_grid(run_dir, rets, factors, lags_alpha, start, end),
            subperiods=_subperiods(rets, rf, lags_rel),
            groups=_group_table(rets, selection, expected_sign, lags_rel),
            universe_sensitivity=None,
        ),
        definitions=DEFINITIONS,
    )
    sc = _clean(sc)
    if write:
        (run_dir / RunFiles.scorecard_strategy).write_text(json.dumps(sc, indent=1))
    return sc
