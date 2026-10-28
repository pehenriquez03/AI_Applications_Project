"""Validation layer: consistency checks, period parsing, adaptation to an ExperimentSpec."""
import json
from pathlib import Path

import pytest

from research_engine.contracts import CheckPassItem, CheckPassResult, StrategySpec
from research_engine.validation.adapt import adapt
from research_engine.validation.consistency import check_consistency, parse_months, group_direction

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def spec():
    return StrategySpec.model_validate_json((FIX / "mini_spec.json").read_text())


@pytest.mark.parametrize("text,months", [
    ("1 month of daily data", 1), ("six months", 6), ("6-month", 6), ("none", 0), ("52 weeks", 12),
    ("3 years of weekly returns", 36), ("t-12 to t-2", 11), ("monthly", 1), ("the previous month", 1),
])
def test_parse_months(text, months):
    assert parse_months(text) == pytest.approx(months)


def test_parse_months_not_stated():
    assert parse_months("not stated") is None and parse_months("whenever") is None


def test_group_direction():
    assert group_direction("lowest MAX decile") == "lowest"
    assert group_direction("top 30% (winners)") == "highest"


def test_consistency_clean(spec):
    tags = {i["tag"] for i in check_consistency(spec)}
    assert "missing_input" not in tags and "groups_identical" not in tags
    assert "adaptable_input" in tags          # market cap used only for value weights


def test_consistency_flags(spec):
    s = spec.model_copy(deep=True)
    s.disfavored_group.value = s.favored_group.value
    s.holding_period.value = "until the cows come home"
    s.expected_sign = "favored_group_lower"
    tags = {i["tag"] for i in check_consistency(s)}
    assert {"groups_identical", "unparseable_period", "sign_inconsistent"} <= tags


def test_adapt_fixture_ready(spec):
    e = adapt(spec, "run1", "a" * 64, None)
    assert e.status == "ready" and e.blocked_reasons == []
    assert e.strategy_slug == "max"
    assert [o.op for o in e.signal_ops] == ["rolling_max_daily_return"]
    assert e.signal_ops[0].params["window_months"] == 1
    assert e.selection["sort"] == "deciles" and e.selection["n_groups"] == 10
    assert e.selection["favored"] == "lowest"
    assert e.holding_months == 1 and e.skip_months == 0 and e.weighting == "equal"
    w = [a for a in e.adaptations if a.field == "weighting"][0]
    assert w.status == "adapted" and w.paper_value == "value"
    assert e.n_adapted >= 3 and e.price_floor == 5.0


def test_adapt_missing_input_blocks(spec):
    s = spec.model_copy(deep=True)
    s.required_inputs.value = ["fundamentals"]
    e = adapt(s, "run2", "b" * 64, None)
    assert e.status == "blocked"
    assert any("required input not in snapshot" in r for r in e.blocked_reasons)


def test_adapt_unknown_signal_blocks(spec):
    s = spec.model_copy(deep=True)
    s.signal_formula_words.value = "ratio of gross profits to total assets"
    s.signal_formula_symbols.value = "GP/A"
    e = adapt(s, "run3", "c" * 64, None)
    assert e.status == "blocked"
    assert "signal maps to no registry operation: ratio of gross profits to total assets" in e.blocked_reasons


def test_adapt_cannot_implement_blocks_unless_disputed(spec):
    s = spec.model_copy(deep=True)
    s.can_implement = False
    s.cannot_implement_reasons = ["needs futures"]
    assert adapt(s, "r", "d" * 64, None).status == "blocked"
    dispute = CheckPassResult(agrees=False, items=[CheckPassItem(
        field="can_implement", issue="the paper uses only stock returns", contradicting_quote="x", page=1)])
    assert adapt(s, "r", "d" * 64, None, check_results=[dispute]).status == "ready"


@pytest.mark.parametrize("words,slug,ops", [
    ("cumulative return over the past six months", "momentum", ["cum_return"]),
    ("return in the same calendar month one year earlier", "seasonality", ["lagged_month_return"]),
    ("idiosyncratic volatility relative to the Fama-French three-factor model", "lowvol", ["residual_vol"]),
    ("ratio of current price to the 52-week high", "high52", ["price_to_rolling_high"]),
    ("information discreteness among past winners", "frog", ["cum_return", "info_discreteness"]),
    ("sensitivity of stock returns to oil price changes", "oil", ["oil_beta"]),
    ("volatility of weekly returns over the past 3 years", "lowvol", ["rolling_vol"]),
])
def test_signal_rule_table(spec, words, slug, ops):
    s = spec.model_copy(deep=True)
    s.signal_formula_words.value = words
    s.signal_formula_symbols.value = "not stated"
    e = adapt(s, "r", "e" * 64, None)
    assert e.strategy_slug == slug and [o.op for o in e.signal_ops] == ops


def test_frog_double_sort(spec):
    s = spec.model_copy(deep=True)
    s.signal_formula_words.value = "information discreteness (ID) of past returns"
    s.formation_window.value = "t-12 to t-2"
    s.holding_period.value = "6 months"
    e = adapt(s, "r", "f" * 64, None)
    ds = e.selection["double_sort"]
    assert ds["outer_sort"] == "quintiles" and ds["outer_favored"] == "highest"
    assert ds["inner_sort"] == "terciles" and ds["inner_favored"] == "lowest"
    assert e.signal_ops[0].params == {"months": 11, "skip": 1, "name": "pret_12_2"}
    assert e.holding_months == 6 and e.min_names == 15
