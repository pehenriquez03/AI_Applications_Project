"""Fund-level Monte Carlo economics.

For each bootstrapped 5-year path the strategy and the equal-weight universe
benchmark are each run through the same fee engine (same FeeConfig). The
manager's incremental fee income is (fees on the strategy fund) - (fees on the
EW-universe fund), discounted at fund["discount_rate"], and NPV is that minus
the research cost. Investor outcomes (ending wealth, drawdown) are reported
separately from manager fee income.

Conventions (all documented in summary["assumptions"]):
  * Wealth is a multiple of starting AUM. end_wealth_strategy / end_wealth_ew
    are NET of fees; end_wealth_spy is GROSS (SPY is a reference, no fee).
  * beats_spy / beats_ew compare net strategy wealth to gross SPY / net EW.
  * Fees accrue daily and crystallize per FeeConfig; each crystallization
    year's total fees (management accrued in the year + performance fee at its
    end) is discounted by (1 + r) ** -year with year = 1, 2, ... (year-end).
  * Calendar: a synthetic calendar with 252 trading days per year (the bootstrap
    has no real dates); hurdle accrual uses calendar days / 365 on it.
  * Common random numbers: the date sequences depend only on (seed, n_obs,
    horizon, mean block). The demeaned run reuses the raw run's dates.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from research_engine.contracts import ECON_BENCHMARK

from .bootstrap import bootstrap_index_matrix
from .fees import FeeConfig, apply_fees_fast, fee_config_from

DAYS_PER_YEAR = 252
QUANTILES = (5, 25, 50, 75, 95)
BLOCK_SENS_PATHS = 300


def trading_calendar(n_points: int, start: str = "2030-01-01") -> pd.DatetimeIndex:
    """Synthetic calendar with ~252 points per calendar year (day i at
    start + round(i * 365 / 252) days)."""
    offs = np.rint(np.arange(n_points) * 365.0 / DAYS_PER_YEAR).astype(int)
    return pd.DatetimeIndex(pd.Timestamp(start) + pd.to_timedelta(offs, unit="D"))


def _maxdd(net_aum: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(net_aum, axis=1)
    return (net_aum / peak - 1.0).min(axis=1)  # negative number


def _fund_fees(rets: np.ndarray, cfg: FeeConfig, dates: pd.DatetimeIndex) -> dict:
    out = apply_fees_fast(rets, dates, cfg)
    out["period_total"] = out["mgmt"] + out["perf"]
    out["total"] = out["period_total"].sum(axis=1)
    return out


def _pv_factors(n_periods: int, disc: float) -> np.ndarray:
    return (1.0 + disc) ** -np.arange(1, n_periods + 1, dtype=float)


def run_economics(returns: pd.DataFrame, fund: dict, research_cost: dict, mc: dict,
                  llm_cost_usd: float, seed) -> tuple[dict, pd.DataFrame]:
    """Monte Carlo economics for one strategy. See module docstring.

    returns: daily DataFrame with strategy_net, ew_universe, spy (NaN rows dropped).
    seed: base seed; defaults to mc["seed"] when None.
    Returns (summary, per-path DataFrame).
    """
    if seed is None:
        seed = mc["seed"]
    cols = ["strategy_net", ECON_BENCHMARK, "spy"]
    df = returns[cols].dropna()
    n_obs = len(df)
    if n_obs < 2:
        raise ValueError("Need at least 2 overlapping return observations.")
    strat = df["strategy_net"].to_numpy(float)
    ew = df[ECON_BENCHMARK].to_numpy(float)
    spy = df["spy"].to_numpy(float)

    cfg = fee_config_from(fund)
    years = int(fund["horizon_years"])
    horizon = years * DAYS_PER_YEAR
    disc = float(fund["discount_rate"])
    n_paths = int(mc["n_paths"])
    mean_block = float(mc["mean_block_days"])
    dates = trading_calendar(horizon + 1)

    research_total = (float(research_cost["dev_review_hours"]) * float(research_cost["hourly_rate"])
                      + float(research_cost["maintenance_hours_per_year"]) * years * float(research_cost["hourly_rate"])
                      + float(llm_cost_usd))

    def evaluate(idx: np.ndarray, strat_r: np.ndarray, with_spy: bool, ew_fees: dict | None = None):
        fs = _fund_fees(strat_r[idx], cfg, dates)
        fe = ew_fees if ew_fees is not None else _fund_fees(ew[idx], cfg, dates)
        pv = _pv_factors(fs["period_total"].shape[1], disc)
        inc_pv = (fs["period_total"] - fe["period_total"]) @ pv
        res = dict(fs=fs, fe=fe, incremental_fees_pv=inc_pv, npv=inc_pv - research_total)
        if with_spy:
            res["spy_end"] = np.prod(1.0 + spy[idx], axis=1)
        return res

    idx = bootstrap_index_matrix(n_obs, n_paths, horizon, mean_block, seed)
    raw = evaluate(idx, strat, True)
    fs, fe = raw["fs"], raw["fe"]
    A = cfg.starting_aum

    paths = pd.DataFrame({
        "end_wealth_strategy": fs["net_aum"][:, -1] / A,
        "end_wealth_ew": fe["net_aum"][:, -1] / A,
        "end_wealth_spy": raw["spy_end"],
        "maxdd_strategy": _maxdd(fs["net_aum"]),
        "fees_strategy": fs["total"],
        "fees_ew": fe["total"],
        "incremental_fees_pv": raw["incremental_fees_pv"],
        "npv": raw["npv"],
    })
    paths["beats_spy"] = paths["end_wealth_strategy"] > paths["end_wealth_spy"]
    paths["beats_ew"] = paths["end_wealth_strategy"] > paths["end_wealth_ew"]
    paths.index.name = "path"

    # Demeaned variant: remove the strategy's historical mean excess over the EW fund
    hist_excess_mean = float(np.mean(strat - ew))
    dem = evaluate(idx, strat - hist_excess_mean, False, ew_fees=fe)
    paths["npv_demeaned"] = dem["npv"]
    paths["incremental_fees_pv_demeaned"] = dem["incremental_fees_pv"]

    # Block-length sensitivity (300 paths, raw returns), fresh dates per block length
    block_sens = {}
    for b in mc["block_sensitivity_days"]:
        idx_b = bootstrap_index_matrix(n_obs, BLOCK_SENS_PATHS, horizon, float(b), seed)
        r = evaluate(idx_b, strat, False)
        block_sens[int(b)] = float((r["npv"] > 0).mean())

    quant_cols = ["end_wealth_strategy", "end_wealth_ew", "end_wealth_spy", "maxdd_strategy",
                  "fees_strategy", "fees_ew", "incremental_fees_pv", "npv",
                  "npv_demeaned", "incremental_fees_pv_demeaned"]
    quantiles = {c: {f"p{q:02d}": float(np.percentile(paths[c], q)) for q in QUANTILES} for c in quant_cols}

    exp_inc = float(paths["incremental_fees_pv"].mean())
    exp_inc_se = float(paths["incremental_fees_pv"].std(ddof=1) / np.sqrt(n_paths))
    breakeven = research_total / (exp_inc / A) if exp_inc > 0 else None

    summary = {
        "quantiles": quantiles,
        "frac_beats_spy": float(paths["beats_spy"].mean()),
        "frac_beats_ew": float(paths["beats_ew"].mean()),
        "frac_npv_positive": float((paths["npv"] > 0).mean()),
        "frac_npv_positive_demeaned": float((paths["npv_demeaned"] > 0).mean()),
        "expected_incremental_fees_pv": exp_inc,
        "expected_incremental_fees_pv_se": exp_inc_se,
        "expected_npv": float(paths["npv"].mean()),
        "breakeven_aum": breakeven,
        "research_cost_total": research_total,
        "llm_cost_usd": float(llm_cost_usd),
        "block_sensitivity": block_sens,
        "historical_mean_daily_excess_vs_ew": hist_excess_mean,
        "n_paths": n_paths,
        "n_obs": n_obs,
        "horizon_days": horizon,
        "mean_block_days": mean_block,
        "seed": seed,
        "bootstrap_index_hash": hashlib.sha256(idx.tobytes()).hexdigest()[:16],
        "benchmark": ECON_BENCHMARK,
        "assumptions": [
            "2/20 with a 5% hard hurdle on a long-only equity strategy is an assumption, not an observed fee schedule",
            "no subscriptions or redemptions; AUM is fixed at starting_aum plus returns net of fees",
            "incremental fees are measured versus an equal-weight universe fund charged the same fee schedule, never versus SPY",
            "end_wealth_strategy and end_wealth_ew are net of fees; end_wealth_spy is gross (no fee)",
            "fees are discounted at year-end of each crystallization year; management fees accrue daily and are aggregated per year",
            "break-even AUM scales expected incremental fees linearly with AUM and is reported, not headlined; None when expected incremental fees are not positive",
            "the bootstrap resamples historical days jointly (stationary, geometric blocks); it cannot create regimes absent from the sample",
            "the demeaned run subtracts the historical mean daily excess return over the EW fund from the strategy, testing NPV with no alpha beyond noise structure",
            "synthetic 252-day trading calendar; hurdle accrues on calendar days/365",
        ],
    }
    return summary, paths
