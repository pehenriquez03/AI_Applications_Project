import hashlib
import time

import numpy as np
import pandas as pd

from research_engine.contracts import FUND, MONTE_CARLO, RESEARCH_COST
from research_engine.economics.bootstrap import bootstrap_index_matrix
from research_engine.economics.fund import run_economics

N = 2500


def _returns(alpha=0.0002, seed=0):
    rng = np.random.default_rng(seed)
    ew = rng.normal(0.0004, 0.01, N)
    strat = ew + alpha + rng.normal(0, 0.004, N)
    spy = rng.normal(0.0004, 0.011, N)
    return pd.DataFrame({"strategy_net": strat, "ew_universe": ew, "spy": spy},
                        index=pd.bdate_range("2015-01-01", periods=N))


def test_summary_keys_and_frac_in_unit_interval():
    summ, paths = run_economics(_returns(), FUND, RESEARCH_COST, MONTE_CARLO, 3.0, MONTE_CARLO["seed"])
    for k in ["quantiles", "frac_beats_spy", "frac_beats_ew", "frac_npv_positive",
              "frac_npv_positive_demeaned", "expected_incremental_fees_pv", "breakeven_aum",
              "research_cost_total", "llm_cost_usd", "block_sensitivity", "assumptions"]:
        assert k in summ
    for c in ["end_wealth_strategy", "end_wealth_ew", "end_wealth_spy", "maxdd_strategy", "beats_spy",
              "beats_ew", "fees_strategy", "fees_ew", "incremental_fees_pv", "npv"]:
        assert c in paths.columns
    assert len(paths) == MONTE_CARLO["n_paths"]
    assert 0 < summ["frac_npv_positive"] < 1 or summ["frac_npv_positive"] == 1.0  # alpha is large; allow all-positive
    assert 0 < summ["frac_npv_positive_demeaned"] < 1
    assert set(summ["block_sensitivity"]) == set(MONTE_CARLO["block_sensitivity_days"])
    for q in summ["quantiles"].values():
        assert list(q) == ["p05", "p25", "p50", "p75", "p95"]
    expected_cost = 80 * 150 + 10 * 5 * 150 + 3.0
    assert summ["research_cost_total"] == expected_cost
    assert summ["breakeven_aum"] is not None and summ["breakeven_aum"] > 0
    assert (paths["maxdd_strategy"] <= 0).all()


def test_frac_npv_positive_strictly_inside_with_weak_alpha():
    summ, _ = run_economics(_returns(alpha=0.00005, seed=2), FUND, RESEARCH_COST, MONTE_CARLO, 0.0, 1)
    assert 0 < summ["frac_npv_positive"] < 1


def test_breakeven_none_for_underperformer():
    df = _returns()
    df["strategy_net"] = df["ew_universe"] - 0.001
    summ, paths = run_economics(df, FUND, RESEARCH_COST, MONTE_CARLO, 0.0, 5)
    assert summ["breakeven_aum"] is None
    assert summ["expected_incremental_fees_pv"] <= 0
    assert summ["frac_npv_positive"] < 0.05


def test_common_random_numbers_across_strategies():
    a, b = _returns(seed=0), _returns(alpha=0.001, seed=99)
    b["strategy_net"] = b["strategy_net"] * 3  # a different strategy, same sample length
    sa, _ = run_economics(a, FUND, RESEARCH_COST, MONTE_CARLO, 0.0, 77)
    sb, _ = run_economics(b, FUND, RESEARCH_COST, MONTE_CARLO, 0.0, 77)
    assert sa["bootstrap_index_hash"] == sb["bootstrap_index_hash"]
    idx = bootstrap_index_matrix(N, MONTE_CARLO["n_paths"], FUND["horizon_years"] * 252,
                                 MONTE_CARLO["mean_block_days"], 77)
    assert sa["bootstrap_index_hash"] == hashlib.sha256(idx.tobytes()).hexdigest()[:16]
    sc, _ = run_economics(a, FUND, RESEARCH_COST, MONTE_CARLO, 0.0, 78)
    assert sc["bootstrap_index_hash"] != sa["bootstrap_index_hash"]


def test_deterministic_same_seed():
    a = run_economics(_returns(), FUND, RESEARCH_COST, MONTE_CARLO, 1.0, 5)
    b = run_economics(_returns(), FUND, RESEARCH_COST, MONTE_CARLO, 1.0, 5)
    pd.testing.assert_frame_equal(a[1], b[1])
    assert a[0]["frac_npv_positive"] == b[0]["frac_npv_positive"]


def test_incremental_fees_identity_and_timing():
    t = time.time()
    summ, paths = run_economics(_returns(), FUND, RESEARCH_COST, MONTE_CARLO, 0.0, 3)
    assert time.time() - t < 90
    np.testing.assert_allclose(paths["npv"], paths["incremental_fees_pv"] - summ["research_cost_total"])
    # undiscounted incremental fees bound the PV (discount rate > 0, same sign structure)
    assert abs(paths["incremental_fees_pv"]).max() <= abs(paths["fees_strategy"] - paths["fees_ew"]).max() + 1e-6
