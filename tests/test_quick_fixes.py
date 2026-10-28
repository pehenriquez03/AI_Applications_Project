"""Offline regressions for the professor-review reporting corrections."""
import json
from pathlib import Path

import pandas as pd

from research_engine.contracts import StrategySpec
from research_engine.evaluation.strategy import _group_table
from research_engine.reporting.results import assemble_results
from research_engine.reporting.rubric import recommend
from research_engine.reporting.narrative import write_narrative
from research_engine.validation.adapt import adapt

RUNS = Path(__file__).resolve().parents[1] / "runs"


def test_momentum_cost_check_uses_saved_benchmark_and_missing_subperiods_are_unknown():
    result = assemble_results(RUNS / "demo_momentum")
    assert result["robustness"]["holds_at_bps"]["10"] is True
    assert result["robustness"]["holds_at_bps"]["25"] is False
    assert all(x is None for x in result["robustness"]["holds_in_subperiod"].values())
    report = recommend(result)
    assert report["rubric"]["flags"]["robustness"] == "unknown"
    reasons = " ".join(report["reasons"])
    assert "10 bps: holds" in reasons and "2 not evaluated" in reasons
    assert "fails at 10 bps" not in reasons and "0 of 2 sub-periods" not in reasons


def test_frog_evaluates_the_group_actually_traded():
    run = RUNS / "demo_frog"
    returns = pd.read_parquet(run / "returns.parquet")
    spec = json.loads((run / "experiment_spec.json").read_text())
    groups = _group_table(returns, spec["selection"], spec["expected_sign"], spec["holding_months"])
    assert returns.strategy_net.equals(returns.group_1)
    assert groups["favored_group"] == 1
    assert groups["expected_rho_sign"] == -1
    assert groups["rho_in_expected_direction"] is False
    assert groups["favored_minus_disfavored"]["mean_monthly_diff"] < 0


def test_frog_skip_log_matches_the_signal_operation():
    spec = StrategySpec.model_validate_json((RUNS / "demo_frog" / "spec_raw_1.json").read_text())
    exp = adapt(spec, "test", "test")
    entry = next(e for e in exp.adaptations if e.field == "skip_window")
    assert exp.skip_months == exp.signal_ops[0].params["skip_months"] == 1
    assert entry.our_value == "1 months"


def test_annual_rebalance_is_labeled_adapted():
    spec = StrategySpec.model_validate_json((RUNS / "demo_decoy_profitability" / "spec_raw_1.json").read_text())
    exp = adapt(spec, "test", "test")
    entry = next(e for e in exp.adaptations if e.field == "rebalance_frequency")
    assert entry.status == "adapted" and entry.our_value == "monthly"


def test_oil_reports_actual_selection_and_trading_coverage():
    result = assemble_results(RUNS / "demo_oil")
    con = result["construction"]
    assert con["sample_month"] == "2025-03-31"
    assert con["last_rebalance"] == "2025-04-01"
    assert "holdings carry forward" in con["coverage_note"]


def test_explicit_template_refresh_ignores_old_model_text(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No model calls during a deterministic refresh")
    monkeypatch.setattr("research_engine.extraction.client.call_structured", forbidden)
    stale = {"summary": "Stale model summary", "sections": {}, "source": "session"}
    (tmp_path / "narrative.json").write_text(json.dumps(stale))
    (tmp_path / "narrative_raw.json").write_text(json.dumps(stale))
    result = assemble_results(RUNS / "demo_momentum")
    rec = recommend(result)
    result.update(recommendation=rec["recommendation"], recommendation_reasons=rec["reasons"], rubric=rec["rubric"])
    narrative = write_narrative(tmp_path, result, llm="template")
    assert narrative["source"] == "template"
    assert "Stale model" not in narrative["summary"]
    assert "numeric tokens only" in narrative["verification_scope"]
    assert "Across 2 extraction runs" in narrative["sections"]["How reliable the interpretation was"]
