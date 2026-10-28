"""Overlapping holding cohorts (Jegadeesh-Titman K-month holding periods).

Implemented cohort convention:
- Every formation month m yields one cohort: the selected names, equal-weighted inside the
  cohort. A cohort lives for K execution days: those of months m, m+1, ..., m+K-1.
- Cohorts occupy K slots (slot = formation-month index mod K). The cohort formed in month m
  replaces the cohort formed in month m-K, so "the expiring cohort funds the new cohort".
- Net target at the execution day of month m = (1/K) x sum over live cohorts of that
  cohort's equal-weight vector. A name held in several cohorts gets the sum of its cohort
  weights. Only the netted vector is traded.
- Start-up: during the first K-1 execution days fewer than K cohorts exist. Each cohort is
  still allotted 1/K of capital and the rest is meant to sit in cash, so the net target sums
  to n_live/K. NOTE: backtest/engine.py renormalises every target to sum to 1 (`w /= tot`),
  so in the current engine the start-up months are in fact fully invested, spread over the
  live cohorts. Keeping the cash needs an engine option (e.g. `normalize=False`, cash = 1 -
  sum of weights); this module already emits the un-normalised weights.
- A month whose cohort is skipped (fewer than min_names selected) leaves its slot untouched:
  the cohort it would have replaced is held one more month (the K=1 analogue of "previous
  holdings carry forward"). Logged.
- A name that is no longer eligible at formation date m is dropped from every live cohort at
  the execution day of month m; the cohort is re-equal-weighted over its survivors and keeps
  its 1/K. A cohort with no survivors frees its slot. Logged once per (cohort, name).
- skip_months affects only the formation signal (handled by the ops); cohort timing is not
  shifted by it.

Approximation. The exact self-financing convention would let each cohort's value drift with
its own returns and trade only when the expiring cohort is replaced (one cohort out, one in,
turnover about 1/K per month). Resetting every live cohort to exactly 1/K (and every name to
1/n inside its cohort) at each execution day approximates that NAV accounting without
tracking per-cohort values inside the engine. The price is extra "drift-correction" turnover:
the engine trades the net change from the drifted book to the reset book, i.e. about 1/K plus
the drift that accumulated over the month. tests/test_cohorts.py measures both pieces.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("research_engine.backtest.cohorts")
log.setLevel(logging.INFO)


@dataclass
class _Cohort:
    formation_date: pd.Timestamp
    names: pd.Index


def net_cohort_targets(
    formation_dates: Sequence[pd.Timestamp],
    execution_days: pd.Series,
    cohorts: dict[pd.Timestamp, Optional[pd.Index]],
    holding_months: int,
    eligible_fn: Callable[[pd.Timestamp], pd.Index],
    label: str = "strategy",
) -> tuple[dict[pd.Timestamp, pd.Series], list[dict]]:
    """Net overlapping cohorts into one target vector per execution day.

    formation_dates: ordered formation month-ends; execution_days: formation date -> execution
    day; cohorts: formation date -> selected names (None or empty = month skipped);
    eligible_fn(formation_date) -> names eligible at that date (data through that date only).
    Returns (targets execution_day -> weights summing to n_live/K, event log).
    """
    K = int(holding_months)
    if K < 1:
        raise ValueError("holding_months must be >= 1")
    slots: list[Optional[_Cohort]] = [None] * K
    targets: dict[pd.Timestamp, pd.Series] = {}
    events: list[dict] = []

    def note(**e):
        e["label"] = label
        events.append(e)
        log.info("cohort %s", " ".join(f"{k}={v}" for k, v in e.items()))

    for i, d in enumerate(formation_dates):
        d = pd.Timestamp(d)
        exec_day = pd.Timestamp(execution_days[d])
        s = i % K
        new = cohorts.get(d)
        if new is not None and len(new):
            slots[s] = _Cohort(d, pd.Index(np.sort(np.asarray(new, dtype=str)), name="ticker"))
        elif slots[s] is not None:
            note(event="carry_forward", formation_date=d.date(), execution_day=exec_day.date(),
                 held_cohort=slots[s].formation_date.date(),
                 reason="no cohort formed this month; the cohort it would replace is held one more month")
        # drop names that are no longer eligible at this formation date
        ok = pd.Index(eligible_fn(d)).astype(str)
        for j, c in enumerate(slots):
            if c is None:
                continue
            gone = c.names.difference(ok)
            if len(gone):
                for t in gone:
                    note(event="drop_ineligible", formation_date=d.date(), execution_day=exec_day.date(),
                         cohort=c.formation_date.date(), ticker=t)
                c.names = c.names.intersection(ok)
                if len(c.names) == 0:
                    slots[j] = None
        live = [c for c in slots if c is not None]
        if not live:
            continue
        w = pd.concat([pd.Series(1.0 / (K * len(c.names)), index=c.names) for c in live])
        w = w.groupby(level=0).sum().sort_index()
        w.index.name = "ticker"
        targets[exec_day] = w
    return targets, events

