"""Fee engine tests: legacy reconciliation, hand-checkable paths, fast-path equality."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_engine.contracts import FUND
from research_engine.economics.fees import (FeeConfig, apply_fees, apply_fees_fast,
                                            fee_config_from, fee_summary)

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy" / "Tests_v2_TrainTest"
LEGACY_CFG = FeeConfig(starting_aum=100_000, management_fee_rate=0.02, performance_fee_rate=0.20,
                       hurdle_rate=0.05, hurdle_type="hard", high_water_mark=True,
                       crystallization_frequency="annual", management_fee_basis="daily_nav", day_count=252)


def _path(daily_returns, start="2020-01-01"):
    r = np.asarray(daily_returns, float)
    eq = np.concatenate([[1.0], np.cumprod(1 + r)])
    return pd.Series(eq, index=pd.bdate_range(start, periods=len(eq)))


def test_fee_config_from_fund():
    cfg = fee_config_from(FUND)
    assert cfg.starting_aum == FUND["starting_aum"]
    assert cfg.management_fee_rate == 0.02 and cfg.performance_fee_rate == 0.20
    assert cfg.hurdle_rate == 0.05 and cfg.hurdle_type == "hard"
    assert cfg.high_water_mark and cfg.crystallization_frequency == "annual"
    assert cfg.day_count == 252 and cfg.management_fee_basis == "daily_nav"


@pytest.mark.parametrize("run", [1, 2, 3, 4, 5])
def test_reproduces_legacy_fee_numbers(run):
    curves = pd.read_csv(LEGACY / "equity_curves" / "equity_curves_rsq_0.0000.csv", index_col=0)
    legacy = pd.read_csv(LEGACY / "fee_economics_by_run.csv")
    row = legacy[(legacy["rsq"] == 0.0) & (legacy["run"] == run)].iloc[0]
    daily, ledger = apply_fees(curves[f"run_{run}"], LEGACY_CFG)
    s = fee_summary(daily, ledger, LEGACY_CFG)
    assert s["management_fees"] == pytest.approx(row["management_fees"], rel=1e-6)
    assert s["performance_fees"] == pytest.approx(row["performance_fees"], rel=1e-6)
    assert s["total_fees"] == pytest.approx(row["total_fees"], rel=1e-6)
    assert s["n_crystallization_periods"] == row["n_crystallization_periods"]


def test_flat_path_pays_only_management_fee():
    n = 504
    cfg = FeeConfig(starting_aum=1_000_000.0)
    daily, ledger = apply_fees(_path(np.zeros(n)), cfg)
    m = 0.02 / 252
    expected = cfg.starting_aum * (1 - (1 - m) ** n)  # NAV decays by (1-m) per day
    assert daily["ManagementFee"].sum() == pytest.approx(expected, rel=1e-12)
    assert daily["PerformanceFee"].sum() == 0.0


def test_performance_fee_year1_none_year2_below_hwm():
    idx = pd.bdate_range("2021-01-01", "2022-12-30")
    y1 = idx.year[1:] == 2021          # return days falling in calendar 2021
    n1, n2 = int(y1.sum()), int((~y1).sum())
    r = np.where(y1, 1.30 ** (1 / n1) - 1, 0.80 ** (1 / n2) - 1)
    eq = pd.Series(np.concatenate([[1.0], np.cumprod(1 + r)]), index=idx)
    cfg = FeeConfig(starting_aum=100.0, management_fee_rate=0.0)  # isolate performance fees
    daily, ledger = apply_fees(eq, cfg)
    assert len(ledger) == 2
    first, second = ledger.iloc[0], ledger.iloc[1]
    assert first["PerformanceFee"] > 0
    assert first["EndNetAUMBeforePerformanceFee"] == pytest.approx(130.0, rel=1e-9)
    assert second["PerformanceFee"] == 0.0
    # year-2 NAV (before fee) is far below the high-water mark set by year 1's net NAV
    assert second["EndNetAUMBeforePerformanceFee"] < second["HighWaterMarkBeforeFee"]


def test_hurdle_blocks_fee_when_return_below_hurdle():
    # exactly 4% over one calendar year, 5% hard hurdle -> no performance fee
    idx = pd.bdate_range("2021-01-01", "2021-12-31")
    n = len(idx) - 1
    r = 1.04 ** (1 / n) - 1
    eq = pd.Series(np.concatenate([[1.0], np.cumprod(np.full(n, 1 + r))]), index=idx)
    cfg = FeeConfig(starting_aum=100.0, management_fee_rate=0.0)
    daily, ledger = apply_fees(eq, cfg)
    assert daily["PerformanceFee"].sum() == 0.0
    # and a 8% year does pay on the excess over the hurdle only
    eq2 = pd.Series(np.concatenate([[1.0], np.cumprod(np.full(n, 1.08 ** (1 / n)))]), index=idx)
    d2, l2 = apply_fees(eq2, cfg)
    hurdle_nav = l2.iloc[0]["HurdleNAV"]
    assert d2["PerformanceFee"].sum() == pytest.approx(0.2 * (108.0 - hurdle_nav), rel=1e-9)


@pytest.mark.parametrize("cfg", [
    FeeConfig(starting_aum=1e8),
    FeeConfig(starting_aum=1e8, hurdle_type="soft"),
    FeeConfig(starting_aum=1e8, high_water_mark=False),
    FeeConfig(starting_aum=1e8, crystallization_frequency="quarterly"),
    FeeConfig(starting_aum=1e8, crystallization_frequency="monthly", management_fee_basis="period_start_nav"),
])
def test_fast_path_equals_apply_fees(cfg):
    rng = np.random.default_rng(7)
    n = 1260
    dates = pd.DatetimeIndex(pd.bdate_range("2030-01-01", periods=n + 1))
    R = rng.normal(0.0005, 0.012, size=(20, n))
    fast = apply_fees_fast(R, dates, cfg)
    for p in range(20):
        eq = pd.Series(np.concatenate([[1.0], np.cumprod(1 + R[p])]), index=dates)
        daily, ledger = apply_fees(eq, cfg)
        np.testing.assert_allclose(fast["net_aum"][p], daily["NetAUM"].to_numpy(), rtol=1e-9, atol=0)
        np.testing.assert_allclose(fast["mgmt"][p], ledger["ManagementFee"].to_numpy(), rtol=1e-9, atol=1e-6)
        np.testing.assert_allclose(fast["perf"][p], ledger["PerformanceFee"].to_numpy(), rtol=1e-9, atol=1e-6)
