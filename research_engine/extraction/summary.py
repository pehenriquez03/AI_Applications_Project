"""Strategy summary for extraction-only runs (dropped PDFs).

One structured model call turns the consensus StrategySpec plus the validation verdict into a
plain-language summary. Without a model (no key, refusal) a deterministic template is written
instead and labeled "template". The answer key is never part of the prompt.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from research_engine.contracts import MATERIAL_FIELDS, RunFiles, StrategySpec

log = logging.getLogger("engine")

SYSTEM_SUMMARY = (
    "You are an investment analyst writing for a portfolio manager. You are given a structured extraction "
    "of a research paper's trading strategy and a verdict on whether our backtest engine (long-only monthly "
    "sorts on daily US equity prices, no fundamentals, no derivatives, no intraday data) can run it. "
    "Write a plain-language summary. Use only what the extraction says; do not invent numbers or results. "
    "If a field is 'not stated', say so rather than guessing."
)


class StrategySummary(BaseModel):
    summary: str = Field(description="4-6 sentences: the signal, how the portfolio is formed and held, "
                                     "which side the paper favors and why, and the paper's headline result")
    data_needs: str = Field(description="1-2 sentences: the inputs the strategy requires")
    implementation_note: str = Field(description="1-2 sentences: whether this engine can run it and what would "
                                                 "have to change or be added if not")


def build_summary_user(spec: StrategySpec, verdict: str, title: str = "") -> str:
    fields = {}
    for f in MATERIAL_FIELDS:
        v = getattr(spec, f, None)
        fields[f] = getattr(v, "value", v)
    body = {"paper_title": title, "can_implement": spec.can_implement,
            "cannot_implement_reasons": spec.cannot_implement_reasons,
            "selected_experiment": spec.selected_experiment, "fields": fields,
            "unresolved_details": spec.unresolved_details, "engine_verdict": verdict}
    return "Extraction and verdict:\n" + json.dumps(body, indent=1, default=str) + \
        "\n\nWrite the summary, the data needs and the implementation note."


def template_summary(spec: StrategySpec, verdict: str) -> dict:
    def val(f):
        v = getattr(spec, f, None)
        v = getattr(v, "value", v)
        if isinstance(v, (list, tuple)):
            v = "; ".join(map(str, v))
        return str(v or "not stated")
    summary = (f"Signal: {val('signal_formula_words')} Portfolios: {val('sort_rule')}, breakpoints {val('breakpoints')}, "
               f"weighting {val('weighting')}, formation {val('formation_window')}, skip {val('skip_window')}, "
               f"held {val('holding_period')}, rebalanced {val('rebalance_frequency')}. Favored group: {val('favored_group')} "
               f"(expected sign: {val('expected_sign')}). Headline result: {val('headline_result')}")
    return {"summary": summary, "data_needs": val("required_inputs"), "implementation_note": verdict,
            "source": "template"}


def write_summary(run_dir: Path, spec: StrategySpec, verdict: str, llm: str = "api",
                  model: Optional[str] = None) -> dict:
    """Write RunFiles.strategy_summary and return it ({summary, data_needs, implementation_note, source})."""
    run_dir = Path(run_dir)
    out = None
    if llm in ("api", "api-refresh"):
        try:
            from research_engine.extraction.client import call_structured
            meta = json.loads((run_dir / RunFiles.paper_meta).read_text()) if (run_dir / RunFiles.paper_meta).exists() else {}
            obj, usage = call_structured(SYSTEM_SUMMARY, build_summary_user(spec, verdict, meta.get("title") or ""),
                                         StrategySummary, tag="summary", run_dir=run_dir, model=model,
                                         refresh=(llm == "api-refresh"))
            out = {**obj.model_dump(), "source": usage.get("source", "api")}
        except Exception as e:  # noqa: BLE001  LLMUnavailable, refusal, parse errors -> template
            log.warning("strategy summary model call unavailable: %s", e)
    if out is None:
        out = template_summary(spec, verdict)
    (run_dir / RunFiles.strategy_summary).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out
