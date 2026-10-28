"""Tests for research_engine.evaluation (stats, strategy scorecard, interpretation scorecard).

Synthetic run folders reuse the real trading calendar and factor file from the
snapshot (read-only); no network or API access.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_engine.contracts import (
    EVAL_END, EVAL_START, MATERIAL_FIELDS, SNAPSHOT_DIR, ExperimentSpec, RunFiles, StrategySpec,
)
from research_engine.evaluation.interpretation import CRITICAL_FIELDS, score_interpretation
from research_engine.evaluation.stats import (
    alpha_regression, annualize, monthly_from_daily, newey_west_tstat, spearman_monotonicity,
)
from research_engine.evaluation.strategy import load_factors, score_strategy

needs_snapshot = pytest.mark.skipif(not (SNAPSHOT_DIR / "ff3_daily.parquet").exists(),
                                    reason="snapshot not available")
N_GROUPS = 10


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lags", [0, 1, 3])
def test_newey_west_matches_statsmodels_hac(lags):
    sm = pytest.importorskip("statsmodels.api")
    x = pd.Series([0.012, -0.004, 0.021, 0.008, -0.013, 0.005, 0.017, 0.002, -0.009, 0.011,
                   0.004, 0.019, -0.006, 0.0, 0.014, -0.002, 0.007, 0.01, -0.011, 0.015])
    mean, t, se = newey_west_tstat(x, lags)
    res = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    assert mean == pytest.approx(x.mean())
    assert se == pytest.approx(float(res.bse[0]), rel=1e-9)
    assert t == pytest.approx(float(res.tvalues[0]), rel=1e-9)


def test_newey_west_known_value_no_lags():
    # lags=0: se = sqrt(sum((x-m)^2)) / n
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    mean, t, se = newey_west_tstat(x, 0)
    assert mean == 2.5
    assert se == pytest.approx(np.sqrt(5.0) / 4)
    assert t == pytest.approx(2.5 / (np.sqrt(5.0) / 4))


def test_spearman_monotonicity():
    rho, p = spearman_monotonicity([0.1, 0.2, 0.3, 0.4, 0.5])
    assert rho == 1.0 and p < 0.05
    assert spearman_monotonicity([5, 4, 3, 2, 1])[0] == -1.0
    assert np.isnan(spearman_monotonicity([1, 1, 1, 1])[0])


def test_annualize_constant_return():
    r = pd.Series(0.0004, index=pd.bdate_range("2020-01-01", periods=252))
    a = annualize(r)
    assert a["cagr"] == pytest.approx(1.0004 ** 252 - 1)
    assert a["max_drawdown"] == 0.0
    assert a["end_wealth"] == pytest.approx(1.0004 ** 252)
    r2 = pd.Series([0.1, -0.5, 0.2], index=pd.bdate_range("2020-01-01", periods=3))
    a2 = annualize(r2)
    assert a2["max_drawdown"] == pytest.approx(-0.5)
    assert a2["drawdown"].min() == pytest.approx(-0.5)
    # risk-free lowers sharpe
    rf = pd.Series(0.0001, index=r.index)
    r3 = r + np.random.default_rng(0).normal(0, 0.005, len(r))
    assert annualize(r3, rf)["sharpe"] < annualize(r3)["sharpe"]


def test_monthly_from_daily_compounds():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-02-03"])
    m = monthly_from_daily(pd.Series([0.1, 0.1, 0.05], index=idx))
    assert list(m.round(6)) == [round(1.1 * 1.1 - 1, 6), 0.05]
    assert list(m.index) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]


@needs_snapshot
def test_alpha_regression_recovers_market_beta():
    f = load_factors(SNAPSHOT_DIR)
    f = f.loc["2015":"2025"]
    rng = np.random.default_rng(1)
    y = 0.0 + 1.0 * f["mkt_rf"] + rng.normal(0, 1e-4, len(f))
    out = alpha_regression(y, f[["mkt_rf"]], lags=5)
    assert out["betas"]["mkt_rf"] == pytest.approx(1.0, abs=0.01)
    assert abs(out["alpha_annual"]) < 0.01
    assert out["r2"] > 0.999


@needs_snapshot
def test_alpha_regression_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    f = load_factors(SNAPSHOT_DIR).loc["2015":"2025"]
    X = f[["mkt_rf", "smb", "hml", "mom"]]
    rng = np.random.default_rng(2)
    y = pd.Series(0.0002 + X.to_numpy() @ np.array([0.9, 0.2, -0.1, 0.05]) + rng.normal(0, 0.004, len(X)), index=X.index)
    out = alpha_regression(y, X, lags=21)
    res = sm.OLS(y.to_numpy(), sm.add_constant(X.to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": 21})
    assert out["alpha_daily"] == pytest.approx(res.params[0])
    assert out["alpha_t"] == pytest.approx(float(res.tvalues[0]), rel=1e-8)
    assert out["betas"]["smb"] == pytest.approx(res.params[2])
    assert out["r2"] == pytest.approx(res.rsquared)


# ---------------------------------------------------------------------------
# synthetic run folder
# ---------------------------------------------------------------------------
def make_run(tmp_path, holding_months=1, favored="lowest", expected_sign="favored_group_higher", seed=0,
             group_direction=-1, write_cost_files=True) -> Path:
    """group_direction=-1: group_1 earns the most (monotone decreasing in group index)."""
    run = tmp_path / "run"
    run.mkdir()
    f = load_factors(SNAPSHOT_DIR)
    idx = f.loc[EVAL_START:EVAL_END].index
    rng = np.random.default_rng(seed)
    n = len(idx)
    ew = pd.Series(f.loc[idx, "mkt_rf"].to_numpy() + f.loc[idx, "rf"].to_numpy() + rng.normal(0, 0.002, n), index=idx)
    spy = ew * 0.95 + rng.normal(0, 0.003, n)
    net = ew + 0.0003 + rng.normal(0, 0.004, n)
    gross = net + 0.00002
    df = pd.DataFrame({"strategy_gross": gross, "strategy_net": net, "spy": spy, "ew_universe": ew})
    for k in range(1, N_GROUPS + 1):
        df[f"group_{k}"] = ew + group_direction * (k - 5.5) * 0.00006 + rng.normal(0, 0.0005, n)
    df.index.name = "date"
    df.to_parquet(run / RunFiles.returns)
    if write_cost_files:
        for bps in (0, 25):
            c = pd.DataFrame({"strategy_net": gross - bps * 1e-6}, index=df.index)
            c.to_parquet(run / f"returns_cost_{bps}.parquet")
        # base (10 bps) must sit between them
        df["strategy_net"] = gross - 10 * 1e-6
        df.to_parquet(run / RunFiles.returns)

    first_days = pd.Series(idx, index=idx).groupby(idx.to_period("M")).first().to_numpy()
    tickers = ["AAPL", "ABT", "A", "AAL", "ACGL"]
    pv = pd.DataFrame({"gross_value": (1 + gross).cumprod(), "net_value": (1 + df["strategy_net"]).cumprod(),
                       "cash": 0.0, "n_holdings": 5, "turnover": np.nan}, index=idx)
    pv.loc[first_days, "turnover"] = 0.3
    pv.loc[first_days[0], "turnover"] = 1.0
    pv.loc[first_days[10], "turnover"] = 0.9       # the max after the initial build
    pv.index.name = "date"
    pv.to_parquet(run / RunFiles.portfolio_value)
    h = pd.DataFrame([dict(date=d, ticker=t, shares=1.0, price=1.0, value=0.2, weight=0.2)
                      for d in first_days for t in tickers])
    h.to_parquet(run / RunFiles.holdings)
    tr = pd.DataFrame([dict(date=d, ticker=t, shares=1.0, price=1.0, value=(0.02 if i % 2 else -0.02), cost=0.00002,
                            reason="rebalance") for d in first_days for i, t in enumerate(tickers)])
    tr.to_parquet(run / RunFiles.trades)

    exp = ExperimentSpec(run_id="t", paper_sha256="x", strategy_slug="max", status="ready", signal_ops=[],
                         selection=dict(sort="deciles", n_groups=N_GROUPS, percent=None, favored=favored, double_sort=None),
                         holding_months=holding_months, expected_sign=expected_sign)
    (run / RunFiles.experiment_spec).write_text(exp.model_dump_json())
    return run


@needs_snapshot
def test_strategy_scorecard_keys_and_values(tmp_path):
    run = make_run(tmp_path)
    sc = score_strategy(run)
    for k in ["meta", "performance", "relative", "beta_vs_spy", "alpha", "spanning", "turnover", "costs",
              "holdings", "capacity", "robustness", "definitions"]:
        assert k in sc, k
    assert (run / RunFiles.scorecard_strategy).exists()
    assert json.loads((run / RunFiles.scorecard_strategy).read_text()) == sc  # JSON round trip, plain floats
    for s in ["strategy_gross", "strategy_net", "spy", "ew_universe"]:
        assert set(sc["performance"][s]) >= {"cagr", "vol", "sharpe", "max_drawdown", "end_wealth"}
    for b in ["vs_spy", "vs_ew_universe"]:
        assert set(sc["relative"][b]) >= {"mean_monthly_diff", "nw_t", "ci95_low", "ci95_high", "annualized_diff"}
        assert sc["relative"][b]["ci95_low"] < sc["relative"][b]["mean_monthly_diff"] < sc["relative"][b]["ci95_high"]
    assert sc["relative"]["vs_ew_universe"]["nw_lags"] == 1
    # constructed edge of 3 bps/day over EW
    assert sc["relative"]["vs_ew_universe"]["nw_t"] > 2
    assert sc["spanning"]["alpha_t"] > 2
    assert sc["spanning"]["alpha_annual"] == pytest.approx(0.0003 * 252, abs=0.03)
    assert sc["alpha"]["alpha_t"] > 2
    assert set(sc["alpha"]["betas"]) == {"mkt_rf", "smb", "hml", "mom"}
    assert sc["alpha"]["nw_lags"] == 21
    assert sc["alpha"]["betas"]["mkt_rf"] == pytest.approx(1.0, abs=0.1)
    assert sc["beta_vs_spy"] == pytest.approx(1 / 0.95, abs=0.15)
    assert sc["performance"]["strategy_net"]["cagr"] > sc["performance"]["ew_universe"]["cagr"]
    assert sc["robustness"]["universe_sensitivity"] is None
    assert "cagr" in sc["definitions"]


@needs_snapshot
def test_scorecard_lags_follow_holding_months(tmp_path):
    sc = score_strategy(make_run(tmp_path, holding_months=3))
    assert sc["relative"]["vs_spy"]["nw_lags"] == 3
    assert sc["alpha"]["nw_lags"] == 63


@needs_snapshot
def test_turnover_costs_holdings(tmp_path):
    sc = score_strategy(make_run(tmp_path))
    assert sc["turnover"]["max_monthly"] == pytest.approx(0.9)     # initial build (1.0) excluded
    assert 0.3 < sc["turnover"]["mean_monthly"] < 0.35
    assert sc["costs"]["total_cost"] == pytest.approx(0.00002 * 5 * 132)
    assert sc["costs"]["pct_of_avg_value"] > 0
    assert sc["holdings"]["mean_n_holdings"] == 5
    assert sc["holdings"]["max_weight"] == pytest.approx(0.2)
    assert sc["holdings"]["mean_hhi"] == pytest.approx(0.2)


@needs_snapshot
def test_capacity_is_positive_and_ordered(tmp_path):
    run = make_run(tmp_path)
    sc = score_strategy(run)
    cap = sc["capacity"]
    assert cap["n_trades"] > 500
    pt = cap["per_trade"]
    assert 0 < pt["mean"] <= pt["p95"] <= pt["max"]
    # double the fund -> double the participation
    cap2 = score_strategy(run, fund_aum=200e6, write=False)["capacity"]
    assert cap2["per_trade"]["mean"] == pytest.approx(2 * pt["mean"], rel=1e-4)


@needs_snapshot
def test_cost_grid_monotone_and_subperiods(tmp_path):
    sc = score_strategy(make_run(tmp_path))
    g = sc["robustness"]["cost_grid"]
    assert set(g) == {"0", "10", "25"}
    assert g["0"]["cagr"] > g["10"]["cagr"] > g["25"]["cagr"]
    assert g["0"]["sharpe"] > g["10"]["sharpe"] > g["25"]["sharpe"]
    assert g["0"]["alpha_t"] > g["25"]["alpha_t"]
    sp = sc["robustness"]["subperiods"]
    assert sp["first"]["end"] < "2020-01-01" <= sp["second"]["start"]
    assert sp["first"]["rel_ew_t"] is not None and sp["second"]["rel_ew_t"] is not None


@needs_snapshot
def test_missing_cost_files_give_null(tmp_path):
    sc = score_strategy(make_run(tmp_path, write_cost_files=False))
    assert sc["robustness"]["cost_grid"]["0"] is None
    assert sc["robustness"]["cost_grid"]["10"] is not None


@needs_snapshot
@pytest.mark.parametrize("favored,sign,direction,expect_rho_sign,expect_ok", [
    ("lowest", "favored_group_higher", -1, -1, True),    # group_1 earns most, favored lowest: rho negative, as expected
    ("lowest", "favored_group_higher", +1, -1, False),   # monotone the wrong way
    ("highest", "favored_group_higher", +1, 1, True),
    ("highest", "favored_group_lower", -1, -1, True),
    ("lowest", "unclear", -1, None, None),
])
def test_group_table_direction(tmp_path, favored, sign, direction, expect_rho_sign, expect_ok):
    sc = score_strategy(make_run(tmp_path, favored=favored, expected_sign=sign, group_direction=direction))
    gt = sc["robustness"]["groups"]
    assert len(gt["rows"]) == N_GROUPS
    assert gt["expected_rho_sign"] == expect_rho_sign
    assert gt["rho_in_expected_direction"] is expect_ok
    assert abs(gt["rho"]) > 0.9
    assert (gt["rho"] > 0) == (direction > 0)
    if favored in ("lowest", "highest") and "favored_minus_disfavored" in gt:
        fm = gt["favored_minus_disfavored"]
        assert fm["favored_group"] == (1 if favored == "lowest" else N_GROUPS)


# ---------------------------------------------------------------------------
# interpretation
# ---------------------------------------------------------------------------
def _ev(v, page=1):
    return dict(value=v, evidence=dict(page=page, quote="q"))


def _spec_dict(can_implement=True) -> dict:
    spec = StrategySpec(
        paper_title="Fixture Paper", authors=["A"], year=2011, selected_experiment="Table 1",
        can_implement=can_implement, cannot_implement_reasons=[] if can_implement else ["needs fundamentals"],
        economic_rationale=_ev("lottery demand"), universe_and_filters=_ev("NYSE stocks"),
        required_inputs=dict(value=["daily prices", "market cap"], evidence=dict(page=1, quote="q")),
        signal_formula_words=_ev("mean of the five highest daily returns in the month"),
        signal_formula_symbols=_ev("MAX = mean(top5 r_d)"),
        formation_window=_ev("1 month of daily data"), skip_window=_ev("none"),
        holding_period=_ev("1 month"), rebalance_frequency=_ev("monthly"),
        sort_rule="deciles", sort_rule_evidence=dict(page=1, quote="q"),
        breakpoints=_ev("equal-count deciles"), favored_group=_ev("lowest MAX decile"),
        disfavored_group=_ev("highest MAX decile"), weighting="equal", weighting_evidence=dict(page=1, quote="q"),
        expected_sign="favored_group_higher", expected_sign_evidence=dict(page=1, quote="q"),
        headline_result=_ev("high-minus-low -1.03% per month"), unresolved_details=["tie handling"],
    )
    return json.loads(spec.model_dump_json())


def _interp_run(tmp_path, spec, with_validation=True) -> Path:
    run = tmp_path / "irun"
    run.mkdir()
    agreement = {f: 1.0 for f in MATERIAL_FIELDS}
    agreement["holding_period"] = 2 / 3
    (run / RunFiles.spec_consensus).write_text(json.dumps(dict(spec=spec, agreement=agreement, n_runs=3)))
    if with_validation:
        ev = {f: dict(valid=True) for f in MATERIAL_FIELDS}
        ev["breakpoints"] = dict(valid=False, reason="quote not found")
        ev["skip_window"] = dict(valid=False)
        (run / "validation.json").write_text(json.dumps(dict(evidence=ev, consistency_issues=["x"], check_pass=[
            dict(agrees=True, items=[]), dict(agrees=False, items=[])], n_check_disagreements=1)))
    (run / RunFiles.adaptation_log).write_text(json.dumps([
        dict(field="a", paper_value=None, our_value="v", status="assumed", reason="r"),
        dict(field="b", paper_value="p", our_value="v", status="adapted", reason="r"),
        dict(field="c", paper_value="p", our_value="v", status="adapted", reason="r"),
        dict(field="d", paper_value=None, our_value="v", status="unresolved", reason="r"),
        dict(field="e", paper_value="p", our_value="p", status="stated", reason="r"),
    ]))
    return run


def test_interpretation_without_reference(tmp_path):
    run = _interp_run(tmp_path, _spec_dict())
    sc = score_interpretation(run, None)
    assert sc["n_runs"] == 3
    assert sc["mean_field_agreement"] == pytest.approx((len(MATERIAL_FIELDS) - 1 + 2 / 3) / len(MATERIAL_FIELDS))
    assert sc["evidence_valid_rate"] == pytest.approx((len(MATERIAL_FIELDS) - 2) / len(MATERIAL_FIELDS))
    assert sc["evidence_invalid_fields"] == ["breakpoints", "skip_window"]
    assert sc["n_check_disagreements"] == 1
    assert (sc["unresolved_count"], sc["assumed_count"], sc["adapted_count"]) == (1, 1, 2)
    assert sc["unresolved_details_count"] == 1
    assert sc["decoy_refused"] is False
    assert sc["reference_label"] is None and sc["agreement_rate"] is None
    assert (run / RunFiles.scorecard_interpretation).exists()


def test_interpretation_decoy_refused(tmp_path):
    sc = score_interpretation(_interp_run(tmp_path, _spec_dict(can_implement=False)), None)
    assert sc["decoy_refused"] is True


def test_interpretation_reference_comparison(tmp_path):
    spec = _spec_dict()
    run = _interp_run(tmp_path, spec)
    ref = copy.deepcopy(spec)
    ref["holding_period"]["value"] = "3 months"                  # critical
    ref["favored_group"]["value"] = "highest MAX decile"         # critical
    ref["breakpoints"]["value"] = "NYSE breakpoints"             # minor
    ref["headline_result"]["value"] = "something else"           # minor
    ref["required_inputs"]["value"] = ["Market Cap", "daily prices"]   # same set, different order/case: still equal
    ref["signal_formula_words"]["value"] = "Mean of the five highest daily returns in the month."  # case/punct only
    refpath = tmp_path / "preliminary_reference.json"
    refpath.write_text(json.dumps(ref))
    sc = score_interpretation(run, refpath)
    assert sorted(sc["critical_errors"]) == ["favored_group", "holding_period"]
    assert sorted(sc["minor_errors"]) == ["breakpoints", "headline_result"]
    assert (sc["n_critical_errors"], sc["n_minor_errors"]) == (2, 2)
    assert sc["n_compared"] == len(MATERIAL_FIELDS)
    assert sc["agreement_rate"] == pytest.approx((len(MATERIAL_FIELDS) - 4) / len(MATERIAL_FIELDS))
    assert sc["reference_label"] == "preliminary_reference.json"
    assert sc["preliminary"] is True
    assert set(sc["critical_errors"]) <= set(CRITICAL_FIELDS)
    # enum field change is detected
    ref2 = copy.deepcopy(spec)
    ref2["sort_rule"] = "quintiles"
    ref2["weighting"] = "value"
    p2 = tmp_path / "reference_spec.json"
    p2.write_text(json.dumps(ref2))
    sc2 = score_interpretation(run, p2)
    assert sc2["critical_errors"] == ["sort_rule"] and sc2["minor_errors"] == ["weighting"]
    assert sc2["preliminary"] is False


def test_interpretation_missing_reference_path_is_treated_as_none(tmp_path):
    run = _interp_run(tmp_path, _spec_dict())
    sc = score_interpretation(run, tmp_path / "nope.json")
    assert sc["reference_label"] is None and sc["agreement_rate"] is None


def test_interpretation_check_disagreements_fall_back_to_count(tmp_path):
    run = _interp_run(tmp_path, _spec_dict())
    v = json.loads((run / "validation.json").read_text())
    del v["n_check_disagreements"]
    (run / "validation.json").write_text(json.dumps(v))
    assert score_interpretation(run, None)["n_check_disagreements"] == 1
