"""Benchmark targets run through the same engine as the strategy."""
from __future__ import annotations

import pandas as pd

from research_engine.contracts import ExperimentSpec

BENCH = "SPY"


def spy_targets(ctx) -> dict[pd.Timestamp, pd.Series]:
    """100% SPY, keyed at the first day of the data; the engine buys it on its first day
    (a target dated before `start` carries into the first day) and never trades again."""
    return {ctx.calendar[0]: pd.Series({BENCH: 1.0})}


def ew_universe_targets(ctx, spec: ExperimentSpec) -> dict[pd.Timestamp, pd.Series]:
    """Equal weight across every eligible name at each formation month-end, executed on the
    next trading day (same dates and eligibility rule as the strategy)."""
    from research_engine.strategies.portfolios import formation_dates
    dates, ex = formation_dates(ctx, spec)
    out = {}
    for d in dates:
        names = ctx.eligible(d, spec.min_history_days, spec.price_floor)
        if len(names):
            out[ex[d]] = pd.Series(1.0 / len(names), index=names)
    return out
