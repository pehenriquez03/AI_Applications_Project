"""Per-run model and backtest window: pipeline.apply_window and the model-keyed LLM cache."""
import pytest

from research_engine.contracts import AdaptationEntry, ExperimentSpec, StrategySpec
from research_engine.extraction.client import cache_key
from research_engine.pipeline import Blocked, apply_window


def _exp() -> ExperimentSpec:
    return ExperimentSpec(run_id="r", paper_sha256="x" * 64, strategy_slug="max", status="ready",
                          signal_ops=[], selection={"sort": "deciles", "n_groups": 10, "favored": "lowest"},
                          adaptations=[AdaptationEntry(field="sample_period", paper_value="1962-2005",
                                                       our_value="2015-01-01..2025-12-31", status="adapted",
                                                       reason="policy")])


def test_apply_window_sets_dates_and_rewrites_the_log_entry():
    e = apply_window(_exp(), "2018-01-01", "2022-12-31")
    assert (e.eval_start, e.eval_end) == ("2018-01-01", "2022-12-31")
    assert e.warmup_start == "2012-01-01"
    sp = next(a for a in e.adaptations if a.field == "sample_period")
    assert sp.our_value.startswith("2018-01-01..2022-12-31") and "chosen for this run" in sp.reason


def test_apply_window_partial_and_invalid():
    e = apply_window(_exp(), None, "2020-12-31")
    assert (e.eval_start, e.eval_end) == ("2015-01-01", "2020-12-31")
    with pytest.raises(Blocked):
        apply_window(_exp(), "2011-01-01", None)          # before the warm-up data
    with pytest.raises(Blocked):
        apply_window(_exp(), "2020-01-01", "2019-12-31")  # empty


def test_cache_key_depends_on_model():
    a = cache_key("s", "u", StrategySpec)
    assert a == cache_key("s", "u", StrategySpec, "claude-sonnet-5-5")   # default model = contracts.MODEL
    assert a != cache_key("s", "u", StrategySpec, "claude-opus-5-5")
