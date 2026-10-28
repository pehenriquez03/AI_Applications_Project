"""Construct + backtest driver shared by the dev run and the pipeline (stages 5-6).

`run_dev(spec, run_dir)` builds targets, runs the strategy, every ranked group and both
benchmarks through the same engine, and writes the run-folder files named in
contracts.RunFiles. `backtest_all(ctx, spec, res, run_dir)` is stage 6 alone.

Run the MAX dev run:  .venv/bin/python -m research_engine.strategies.dev_run
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from research_engine.contracts import COST_BPS_GRID, RUNS_DIR, ExperimentSpec, RunFiles, SignalOp
from research_engine.backtest.benchmarks import ew_universe_targets, spy_targets
from research_engine.backtest.engine import run_backtest
from research_engine.strategies.portfolios import ConstructionResult, build_targets
from research_engine.strategies import registry

log = logging.getLogger("research_engine.backtest")


def eval_window(ctx, spec: ExperimentSpec) -> tuple[pd.Timestamp, pd.Timestamp]:
    cal = ctx.calendar
    days = cal[(cal >= pd.Timestamp(spec.eval_start)) & (cal <= pd.Timestamp(spec.eval_end))]
    return days[0], days[-1]


def compile_plan(spec: ExperimentSpec) -> dict:
    """The executable plan: registry ops with params, selection, holding, conventions."""
    known = set(registry.list_ops())
    ops = [o.model_dump() for o in spec.signal_ops]
    ds = (spec.selection or {}).get("double_sort")
    names = [o["op"] for o in ops] + ([ds["outer_op"], ds["inner_op"]] if ds else [])
    unknown = sorted(set(names) - known)
    if unknown:
        raise KeyError(f"plan references unknown ops: {unknown}")
    return {"signal_ops": ops, "selection": spec.selection, "weighting": spec.weighting,
            "holding_months": spec.holding_months, "min_names": spec.min_names,
            "min_history_days": spec.min_history_days, "price_floor": spec.price_floor,
            "cost_bps": spec.cost_bps, "cost_grid": list(COST_BPS_GRID),
            "eval_start": spec.eval_start, "eval_end": spec.eval_end, "execution": spec.execution,
            "benchmarks": list(spec.benchmarks), "registry_ops": sorted(known)}


def backtest_all(ctx, spec: ExperimentSpec, res: ConstructionResult, run_dir: Path) -> str:
    run_dir = Path(run_dir)
    log.setLevel(logging.INFO)
    t0 = time.time()
    start, end = eval_window(ctx, spec)
    strat = run_backtest(ctx, res.targets, spec.cost_bps, start, end)
    ret = pd.DataFrame({"strategy_gross": strat.returns["gross"], "strategy_net": strat.returns["net"]})
    spy = run_backtest(ctx, spy_targets(ctx), spec.cost_bps, start, end, record=False)
    ew = run_backtest(ctx, ew_universe_targets(ctx, spec), spec.cost_bps, start, end, record=False)
    ret["spy"] = spy.returns["net"]
    ret["ew_universe"] = ew.returns["net"]
    for g in sorted(res.group_targets):
        gr = run_backtest(ctx, res.group_targets[g], spec.cost_bps, start, end, record=False)
        ret[f"group_{g}"] = gr.returns["net"]
    ret.index.name = "date"
    ret.to_parquet(run_dir / RunFiles.returns)
    for c in COST_BPS_GRID:
        r = strat if c == spec.cost_bps else run_backtest(ctx, res.targets, c, start, end, record=False)
        pd.DataFrame({"strategy_gross": r.returns["gross"], "strategy_net": r.returns["net"]}).to_parquet(
            run_dir / f"returns_cost_{c}.parquet")
    strat.holdings.to_parquet(run_dir / RunFiles.holdings)
    strat.trades.to_parquet(run_dir / RunFiles.trades)
    strat.values.to_parquet(run_dir / RunFiles.portfolio_value)
    ctx.eligibility_panel(res.formation_dates, spec.min_history_days, spec.price_floor).to_parquet(
        run_dir / RunFiles.eligibility)

    log.info("construction: %d formation months, %d groups, favored group %d, %d strategy months skipped",
             len(res.formation_dates), res.n_groups, res.favored_group, len(res.skipped_months))
    for s in res.skip_log:
        log.info("skipped %s exec %s group %s: %s", s["formation_date"].date(), s["execution_day"].date(),
                 s["group"], s["reason"])
    for d, n in res.eligible_counts.items():
        log.info("eligible %s %d", d.date(), n)
    for line in strat.log:
        log.info("strategy %s", line)
    fs = strat.forced_sales
    log.info("strategy forced sales: %d; benchmark log lines: spy %d, ew %d", len(fs), len(spy.log), len(ew.log))
    for line in ew.log:
        log.info("ew_universe %s", line)
    for w in data_warnings(ctx, spec):
        log.info("data_warning %s", w)
    msg = (f"{len(ret)} days · net CAGR {_cagr(ret['strategy_net']):.1%} · "
           f"turnover {strat.values['turnover'].mean():.0%}/rebalance · {time.time() - t0:.1f}s")
    log.info("backtest: %s", msg)
    return msg


def data_warnings(ctx, spec: ExperimentSpec, up: float = 1.0, down: float = -0.75) -> list[str]:
    """Daily adjusted-close moves above +100% or below -75% in the sample: usually a Yahoo
    adjustment error (spin-off, special dividend, ticker reuse), listed for the reviewer."""
    r = ctx.returns.loc[pd.Timestamp(spec.warmup_start):pd.Timestamp(spec.eval_end)]
    s = r.stack()
    s = s[(s > up) | (s < down)]
    return [f"{d.date()} {t} daily return {x:+.1%}" for (d, t), x in s.items()]


def _cagr(r: pd.Series) -> float:
    return float((1 + r).prod() ** (252 / len(r)) - 1)


def run_dev(spec: ExperimentSpec, run_dir: Path, ctx=None) -> ConstructionResult:
    """Stages 5-6 end to end into run_dir (writes run.log itself)."""
    from research_engine.data.loaders import DataContext
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(run_dir / RunFiles.run_log, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(handler)
    try:
        ctx = ctx or DataContext.load()
        (run_dir / RunFiles.experiment_spec).write_text(spec.model_dump_json(indent=1))
        plan = compile_plan(spec)
        res = build_targets(ctx, spec)
        plan["skipped_months"] = [str(d.date()) for d in res.skipped_months]
        plan["n_formation_months"] = len(res.formation_dates)
        (run_dir / RunFiles.plan).write_text(json.dumps(plan, indent=1, default=str))
        res.signals_long.to_parquet(run_dir / RunFiles.signals)
        backtest_all(ctx, spec, res, run_dir)
    finally:
        log.removeHandler(handler)
        handler.close()
    return res


def summarize(run_dir: Path) -> list[str]:
    run_dir = Path(run_dir)
    r = pd.read_parquet(run_dir / RunFiles.returns)
    v = pd.read_parquet(run_dir / RunFiles.portfolio_value)

    def ann(s):
        return _cagr(s), float(s.std() * np.sqrt(252))
    lines = []
    for c in ("strategy_net", "spy", "ew_universe"):
        a, s = ann(r[c])
        lines.append(f"{c:<13} ann. return {a:7.2%}  ann. vol {s:6.2%}")
    lines.append(f"mean turnover per monthly rebalance {v['turnover'].mean():.1%}")
    lines.append(f"mean n_holdings {v['n_holdings'].mean():.1f}")
    groups = [c for c in r.columns if c.startswith("group_")]
    monthly = (1 + r[groups]).resample("ME").prod() - 1
    m = monthly.mean()
    lines.append("group mean monthly net return: " + " ".join(f"{k.split('_')[1]}:{x:.2%}" for k, x in m.items()))
    return lines


def max_spec(run_id: str = "dev_max") -> ExperimentSpec:
    return ExperimentSpec(
        run_id=run_id, paper_sha256="dev", strategy_slug="max", status="ready",
        signal_ops=[SignalOp(op="rolling_max_daily_return", params={})],
        selection={"sort": "deciles", "n_groups": 10, "favored": "lowest", "percent": None, "double_sort": None},
        holding_months=1, cost_bps=10, eval_start="2015-01-01", eval_end="2025-12-31",
        warmup_start="2012-01-01", expected_sign="favored_group_higher")


if __name__ == "__main__":
    t = time.time()
    out = RUNS_DIR / "dev_max"
    run_dev(max_spec(), out)
    for line in summarize(out):
        print(line)
    print(f"elapsed {time.time() - t:.1f}s -> {out}")
