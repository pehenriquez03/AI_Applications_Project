"""Hedge-fund fee engine (copied from legacy/lib/fees.py; logic unchanged) plus
a vectorized fast path (`apply_fees_fast`) that is proven equal to `apply_fees`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd


HurdleType = Literal["hard", "soft"]
CrystallizationFrequency = Literal["annual", "quarterly", "monthly"]
ManagementFeeBasis = Literal["daily_nav", "period_start_nav"]


@dataclass(frozen=True)
class FeeConfig:
    """Hedge-fund fee assumptions."""

    starting_aum: float = 100_000_000.0
    management_fee_rate: float = 0.02
    performance_fee_rate: float = 0.20
    hurdle_rate: float = 0.05
    hurdle_type: HurdleType = "hard"
    high_water_mark: bool = True
    crystallization_frequency: CrystallizationFrequency = "annual"
    management_fee_basis: ManagementFeeBasis = "daily_nav"
    day_count: int = 252

    def __post_init__(self):
        if self.starting_aum <= 0:
            raise ValueError("starting_aum must be positive.")
        for name, value in [
            ("management_fee_rate", self.management_fee_rate),
            ("performance_fee_rate", self.performance_fee_rate),
            ("hurdle_rate", self.hurdle_rate),
        ]:
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.hurdle_type not in {"hard", "soft"}:
            raise ValueError("hurdle_type must be 'hard' or 'soft'.")
        if self.crystallization_frequency not in {"annual", "quarterly", "monthly"}:
            raise ValueError("crystallization_frequency must be annual, quarterly, or monthly.")
        if self.management_fee_basis not in {"daily_nav", "period_start_nav"}:
            raise ValueError("management_fee_basis must be daily_nav or period_start_nav.")
        if self.day_count <= 0:
            raise ValueError("day_count must be positive.")


def _normalize_equity_curve(gross_equity: pd.Series) -> pd.Series:
    if not isinstance(gross_equity, pd.Series):
        raise TypeError("gross_equity must be a pandas Series.")

    s = gross_equity.copy()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).normalize()
    s = s.sort_index()

    if s.index.has_duplicates:
        raise ValueError("gross_equity contains duplicate dates.")

    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        raise ValueError("gross_equity is empty after cleaning.")
    if (s <= 0).any():
        raise ValueError("gross_equity must remain strictly positive.")

    return s / float(s.iloc[0])


def _period_key(index: pd.DatetimeIndex, frequency: str) -> pd.PeriodIndex:
    if frequency == "annual":
        return index.to_period("Y")
    if frequency == "quarterly":
        return index.to_period("Q")
    if frequency == "monthly":
        return index.to_period("M")
    raise ValueError(f"Unsupported crystallization frequency: {frequency}")


def _hurdle_for_period(start_date: pd.Timestamp, end_date: pd.Timestamp, annual_hurdle: float) -> float:
    days = max((end_date - start_date).days, 1)
    return (1.0 + annual_hurdle) ** (days / 365.0) - 1.0


def apply_fees(gross_equity: pd.Series, config: FeeConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply fees to one gross equity curve.

    Daily sequence:
      1. Apply gross strategy return to prior net AUM.
      2. Accrue management fee.
      3. If crystallization date, assess performance fee.
      4. Update high-water mark after fee deduction.
    """
    gross = _normalize_equity_curve(gross_equity)
    gross_returns = gross.pct_change().fillna(0.0)
    dates = gross.index
    n = len(dates)

    keys = _period_key(dates, config.crystallization_frequency)
    period_end = np.zeros(n, dtype=bool)
    for i in range(n):
        if i == n - 1 or keys[i] != keys[i + 1]:
            period_end[i] = True

    gross_aum = gross * config.starting_aum
    net_aum = np.full(n, np.nan)
    mgmt_fee = np.zeros(n)
    perf_fee = np.zeros(n)
    hwm_series = np.full(n, np.nan)

    net_aum[0] = config.starting_aum
    high_water_mark = config.starting_aum
    period_start_nav = config.starting_aum
    period_start_date = dates[0]

    ledger_rows = []

    for i in range(1, n):
        nav_before_fees = net_aum[i - 1] * (1.0 + gross_returns.iloc[i])

        if config.management_fee_basis == "daily_nav":
            mgmt_base = nav_before_fees
        else:
            mgmt_base = period_start_nav

        mgmt_fee[i] = mgmt_base * (config.management_fee_rate / config.day_count)
        nav_after_mgmt = nav_before_fees - mgmt_fee[i]

        if nav_after_mgmt <= 0:
            raise ValueError(f"Fees drove NAV non-positive on {dates[i].date()}.")

        if period_end[i]:
            hurdle_return = _hurdle_for_period(period_start_date, dates[i], config.hurdle_rate)
            hurdle_nav = period_start_nav * (1.0 + hurdle_return)

            reference_nav = period_start_nav
            if config.high_water_mark:
                reference_nav = max(reference_nav, high_water_mark)

            if config.hurdle_type == "hard":
                fee_threshold = max(reference_nav, hurdle_nav)
                eligible_profit = max(nav_after_mgmt - fee_threshold, 0.0)
            else:
                clears_hurdle = nav_after_mgmt > hurdle_nav
                eligible_profit = max(nav_after_mgmt - reference_nav, 0.0) if clears_hurdle else 0.0

            perf_fee[i] = config.performance_fee_rate * eligible_profit
            nav_after_perf = nav_after_mgmt - perf_fee[i]

            if nav_after_perf <= 0:
                raise ValueError(f"Performance fee drove NAV non-positive on {dates[i].date()}.")

            mask = (dates >= period_start_date) & (dates <= dates[i])
            period_mgmt_fee = float(mgmt_fee[mask].sum())

            ledger_rows.append({
                "Period": str(keys[i]),
                "StartDate": period_start_date,
                "EndDate": dates[i],
                "StartNetAUM": period_start_nav,
                "EndNetAUMBeforePerformanceFee": nav_after_mgmt,
                "EndNetAUMAfterFees": nav_after_perf,
                "GrossReturn": float(gross.loc[dates[i]] / gross.loc[period_start_date] - 1.0) if dates[i] != period_start_date else 0.0,
                "NetReturn": float(nav_after_perf / period_start_nav - 1.0),
                "HurdleReturn": hurdle_return,
                "HurdleNAV": hurdle_nav,
                "HighWaterMarkBeforeFee": high_water_mark,
                "ManagementFee": period_mgmt_fee,
                "PerformanceFee": perf_fee[i],
                "TotalFee": period_mgmt_fee + perf_fee[i],
            })

            net_aum[i] = nav_after_perf

            if config.high_water_mark:
                high_water_mark = max(high_water_mark, nav_after_perf)

            period_start_nav = nav_after_perf
            if i + 1 < n:
                period_start_date = dates[i + 1]
        else:
            net_aum[i] = nav_after_mgmt

        hwm_series[i] = high_water_mark

    hwm_series[0] = config.starting_aum
    total_fee = mgmt_fee + perf_fee

    daily = pd.DataFrame({
        "GrossEquity": gross,
        "GrossAUM": gross_aum,
        "NetAUM": net_aum,
        "NetEquity": net_aum / config.starting_aum,
        "ManagementFee": mgmt_fee,
        "PerformanceFee": perf_fee,
        "TotalFee": total_fee,
        "CumulativeManagementFee": np.cumsum(mgmt_fee),
        "CumulativePerformanceFee": np.cumsum(perf_fee),
        "CumulativeTotalFee": np.cumsum(total_fee),
        "HighWaterMark": hwm_series,
    }, index=dates)

    return daily, pd.DataFrame(ledger_rows)


def fee_summary(daily: pd.DataFrame, ledger: pd.DataFrame, config: FeeConfig) -> Dict[str, float]:
    total_mgmt = float(daily["ManagementFee"].sum())
    total_perf = float(daily["PerformanceFee"].sum())
    total = total_mgmt + total_perf

    return {
        "starting_aum": config.starting_aum,
        "ending_gross_aum": float(daily["GrossAUM"].iloc[-1]),
        "ending_net_aum": float(daily["NetAUM"].iloc[-1]),
        "management_fees": total_mgmt,
        "performance_fees": total_perf,
        "total_fees": total,
        "fees_as_pct_starting_aum": total / config.starting_aum,
        "performance_fee_crystallizations": int((ledger["PerformanceFee"] > 0).sum()) if not ledger.empty else 0,
        "n_crystallization_periods": int(len(ledger)),
    }


# ----------------------------------------------------------------------------
# Additions for the research engine
# ----------------------------------------------------------------------------
def fee_config_from(fund: dict) -> FeeConfig:
    """Build a FeeConfig from the contracts.FUND dict (extra keys such as
    horizon_years and discount_rate are ignored)."""
    return FeeConfig(
        starting_aum=float(fund["starting_aum"]),
        management_fee_rate=float(fund["management_fee_rate"]),
        performance_fee_rate=float(fund["performance_fee_rate"]),
        hurdle_rate=float(fund["hurdle_rate"]),
        hurdle_type=fund["hurdle_type"],
        high_water_mark=bool(fund["high_water_mark"]),
        crystallization_frequency=fund["crystallization_frequency"],
    )


def apply_fees_fast(gross_returns: np.ndarray, dates: pd.DatetimeIndex, config: FeeConfig) -> Dict[str, np.ndarray]:
    """Vectorized-over-paths equivalent of `apply_fees` (same daily sequence).

    gross_returns: (n_paths, n_dates - 1) daily gross returns for days 1..n-1
    (day 0 is the starting point, exactly as in apply_fees where the first
    return is dropped). dates: the shared calendar (n_dates points).
    Crystallization periods and hurdle returns depend only on the dates, so
    they are computed once and shared by all paths; the loop is over days.

    Returns dict with:
      net_aum  (n_paths, n_dates)   net NAV path (== apply_fees NetAUM)
      mgmt     (n_paths, n_periods) management fees accrued in each period
      perf     (n_paths, n_periods) performance fee crystallized at period end
      period_end_idx (n_periods,)   index into dates of each period end
    """
    R = np.asarray(gross_returns, dtype=float)
    if R.ndim == 1:
        R = R[None, :]
    n_paths, n_ret = R.shape
    n = n_ret + 1
    dates = pd.DatetimeIndex(dates).normalize()
    if len(dates) != n:
        raise ValueError("dates must have one more point than gross_returns has columns.")
    keys = _period_key(dates, config.crystallization_frequency)
    period_end = np.zeros(n, dtype=bool)
    for i in range(n):
        if i == n - 1 or keys[i] != keys[i + 1]:
            period_end[i] = True
    pe_idx = np.flatnonzero(period_end)
    pe_idx_ex0 = pe_idx[pe_idx >= 1]
    # hurdle return per period end (period start = dates[0] or the day after the previous end)
    starts = {}
    start_date = dates[0]
    for i in range(1, n):
        if period_end[i]:
            starts[i] = start_date
            if i + 1 < n:
                start_date = dates[i + 1]
    hurdle = {i: _hurdle_for_period(starts[i], dates[i], config.hurdle_rate) for i in starts}

    A = config.starting_aum
    rate_d = config.management_fee_rate / config.day_count
    net = np.empty((n_paths, n))
    net[:, 0] = A
    nav = np.full(n_paths, A)
    hwm = np.full(n_paths, A)
    psn = np.full(n_paths, A)
    mg_acc = np.zeros(n_paths)
    k_of = {int(i): k for k, i in enumerate(pe_idx_ex0)}
    mgmt = np.zeros((n_paths, len(pe_idx_ex0)))
    perf = np.zeros((n_paths, len(pe_idx_ex0)))
    daily_basis = config.management_fee_basis == "daily_nav"
    for i in range(1, n):
        before = nav * (1.0 + R[:, i - 1])
        m = (before if daily_basis else psn) * rate_d
        after = before - m
        if (after <= 0).any():
            raise ValueError(f"Fees drove NAV non-positive on day {i}.")
        mg_acc += m
        if period_end[i]:
            hurdle_nav = psn * (1.0 + hurdle[i])
            ref = np.maximum(psn, hwm) if config.high_water_mark else psn
            if config.hurdle_type == "hard":
                elig = np.maximum(after - np.maximum(ref, hurdle_nav), 0.0)
            else:
                elig = np.where(after > hurdle_nav, np.maximum(after - ref, 0.0), 0.0)
            pf = config.performance_fee_rate * elig
            after_p = after - pf
            if (after_p <= 0).any():
                raise ValueError(f"Performance fee drove NAV non-positive on day {i}.")
            k = k_of[i]
            mgmt[:, k] = mg_acc
            perf[:, k] = pf
            mg_acc = np.zeros(n_paths)
            nav = after_p
            if config.high_water_mark:
                hwm = np.maximum(hwm, after_p)
            psn = after_p
        else:
            nav = after
        net[:, i] = nav
    return {"net_aum": net, "mgmt": mgmt, "perf": perf, "period_end_idx": pe_idx_ex0}
