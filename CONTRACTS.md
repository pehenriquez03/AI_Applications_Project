# Technical interfaces

Shared schemas and constants live in `research_engine/contracts.py`. This file
describes the module interfaces and run artifact formats.

## Python environment
`.venv/bin/python` in the fork root. Run tests with `.venv/bin/pytest -q`.

## Data access (`research_engine/data/loaders.py`)
```python
ctx = DataContext.load()          # reads SNAPSHOT_DIR, UNIVERSE_FILE
ctx.adj_close, ctx.close, ctx.high, ctx.low, ctx.volume   # DataFrame date x ticker, full 2012-2025
ctx.returns                        # daily simple returns from adj_close
ctx.spy_returns                    # Series
ctx.factors                        # DataFrame: mkt_rf, smb, hml, mom, rf (daily, decimal)
ctx.calendar                       # DatetimeIndex of trading days
ctx.month_ends                     # DatetimeIndex: last trading day of each month
ctx.first_days                     # DatetimeIndex: first trading day of each month (execution days)
ctx.add_dates                      # Series ticker -> Timestamp (S&P 500 add date; NaT if unknown -> treated as always eligible)
ctx.eligible(date, min_history_days, price_floor) -> pd.Index of tickers
```
`ctx.raw_split_adjusted(field)` returns close/high/low adjusted for splits only
(rebuilt from the splits panel), for price-level signals.

## Registry operation signature (`research_engine/strategies/registry.py`)
```python
@register("rolling_max_daily_return")
def rolling_max_daily_return(ctx: DataContext, dates: pd.DatetimeIndex, *, window_months: int = 1, min_days: int = 15) -> pd.DataFrame:
    """Return a DataFrame indexed by `dates` (month-ends), columns = tickers, value = signal.
    MUST use only data with index <= each date (strictly no look-ahead)."""
```
Ops live in `research_engine/strategies/ops/<name>.py` and register on import.
Cross-sectional selection is not an op; `portfolios.py` handles sort/select/weight.

## Construction output (`portfolios.py`)
`build_targets(ctx, spec: ExperimentSpec) -> (signals_long: DataFrame, targets: dict[Timestamp, Series])`
- `signals_long` columns: date, ticker, signal, group (1..N, 1 = lowest signal), selected (bool), weight, cohort (formation date)
- `targets[execution_day]` = Series ticker -> weight summing to 1 (netted across cohorts)
- Also returns group memberships for every group so the engine can run all groups.

## Backtest engine (`backtest/engine.py`)
`run_backtest(ctx, targets: dict[Timestamp, Series], cost_bps: int, start, end) -> BacktestResult`
with `.returns` (daily gross and net Series), `.values` (DataFrame gross_value, net_value, cash, n_holdings),
`.holdings` (long), `.trades` (long). Execution at the target's day close; mark to market daily;
forced sale at last price when a holding leaves the eligible universe.
Benchmarks: `run_backtest` on a single-asset target (SPY) and on equal-weight-all-eligible monthly targets.

## returns.parquet (written by the pipeline after stage 6)
Wide daily DataFrame over EVAL_START..EVAL_END: `strategy_gross, strategy_net, spy, ew_universe, group_1 .. group_N` (group_k net at COST_BPS_BASE).
Plus `returns_cost_{bps}.parquet` for the cost grid (strategy_net only).

## Economics interface (`economics/fund.py`)
`run_economics(returns: DataFrame, fund: dict, research_cost: dict, mc: dict, llm_cost_usd: float, seed) -> (summary: dict, paths: DataFrame)`
Input columns used: `strategy_net`, `ew_universe`, `spy`. Joint stationary bootstrap. Outputs per path:
`end_wealth_strategy, end_wealth_ew, end_wealth_spy, maxdd_strategy, beats_spy, beats_ew, fees_strategy, fees_ew, incremental_fees_pv, npv`.
Summary keys: quantiles of each, `frac_beats_spy, frac_beats_ew, frac_npv_positive, frac_npv_positive_demeaned, expected_incremental_fees_pv, breakeven_aum (None if nonpositive), research_cost_total, block_sensitivity: {days: frac_npv_positive}`.

## Evaluation interface (`evaluation/strategy.py`)
`score_strategy(run_dir) -> dict` reads returns.parquet, holdings, trades, factors; writes scorecard_strategy.json.
`score_interpretation(run_dir, reference_spec_path | None) -> dict` writes scorecard_interpretation.json.

## results.json (assembled by `reporting/results.py`)
Top-level keys: paper, extraction, adaptation, data, construction, performance, robustness,
economics, interpretation, rubric, recommendation, limitations, llm_usage.
Every number the narrative cites must appear somewhere in this JSON.

## Progress
Pipeline stages call `progress.update(n, status, detail)` which rewrites `progress.json`.
The app polls `progress.json` every second while a run is active.

## LLM calls
Only via `research_engine/extraction/client.py`: `call_structured(system, user, schema: type[BaseModel], tag) -> (obj, usage)`.
Cache key = sha256(model | PROMPT_VERSION | SCHEMA_VERSION | system | user | schema name). Cache dir CACHE_DIR.
Key from `.env` (ANTHROPIC_API_KEY). If the key is missing or rejected, raise `LLMUnavailable` with a clear message;
callers decide whether to fall back to cache/fixtures.
