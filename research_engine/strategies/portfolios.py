"""Turn an ExperimentSpec into dated signals, group memberships and target weights.

Portfolio construction conventions:
- Formation at each month-end m using data through m's close; execution at the first
  trading day after m (ctx.execution_day(m), a member of ctx.first_days).
- Groups are equal-count sorts of the eligible names with a non-missing signal at m;
  group 1 = lowest signal. Ties are broken by ticker so results are deterministic.
- A selected group with fewer than spec.min_names names is skipped for that month: no
  new target is issued (the previous holdings carry forward) and the skip is logged.
- Equal weights. holding_months = K > 1 nets K overlapping cohorts into one target per
  execution day, each cohort at 1/K (strategies/cohorts.py documents the rule, the start-up
  months and the approximation). group_targets follow the same cohort rule. signals_long is
  per formation month either way: cohort = formation date, weight = the cohort-internal
  weight 1/n_selected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from research_engine.contracts import TOP_PERCENT_DEFAULT, ExperimentSpec
from research_engine.strategies import registry
from research_engine.strategies.cohorts import net_cohort_targets

SORT_GROUPS = {"deciles": 10, "quintiles": 5, "terciles": 3}


@dataclass
class ConstructionResult:
    signals_long: pd.DataFrame                      # date, ticker, signal, group, selected, weight, cohort (+ outer_* for double sorts)
    targets: dict[pd.Timestamp, pd.Series]          # execution day -> weights (sum 1)
    group_targets: dict[int, dict[pd.Timestamp, pd.Series]]
    eligible_counts: pd.Series                      # formation month-end -> number of eligible names
    skipped_months: list[pd.Timestamp] = field(default_factory=list)   # strategy formation dates skipped
    skip_log: list[dict] = field(default_factory=list)                 # every skip, strategy and groups
    n_groups: int = 0
    favored_group: int = 0
    formation_dates: pd.DatetimeIndex = None
    execution_days: pd.Series = None                # formation date -> execution day
    holding_months: int = 1
    cohort_log: list[dict] = field(default_factory=list)               # carry-forwards and ineligible drops (K > 1)


# --------------------------------------------------------------------------- helpers
def formation_dates(ctx, spec: ExperimentSpec) -> tuple[pd.DatetimeIndex, pd.Series]:
    """Month-ends whose execution day falls inside [eval_start, eval_end]."""
    me = ctx.month_ends
    ex = pd.Series([ctx.execution_day(d) for d in me], index=me, dtype="datetime64[ns]")
    keep = ex.notna() & (ex >= pd.Timestamp(spec.eval_start)) & (ex <= pd.Timestamp(spec.eval_end))
    ex = ex[keep]
    ex.index.name = "date"
    return pd.DatetimeIndex(ex.index, name="date"), ex


def assign_groups(values: pd.Series, n_groups: int) -> pd.Series:
    """Equal-count groups 1..n_groups (1 = lowest); ties broken by ticker."""
    v = values.dropna()
    order = np.lexsort((v.index.to_numpy().astype(str), v.to_numpy()))
    g = np.empty(len(v), dtype=np.int64)
    g[order] = np.arange(len(v)) * n_groups // max(len(v), 1) + 1
    return pd.Series(g, index=v.index)


def assign_percent_groups(values: pd.Series, percent: float) -> pd.Series:
    """Three groups: 1 = bottom `percent`, 3 = top `percent`, 2 = the middle."""
    v = values.dropna()
    n = len(v)
    k = int(round(percent * n))
    order = np.lexsort((v.index.to_numpy().astype(str), v.to_numpy()))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    g = np.where(rank < k, 1, np.where(rank >= n - k, 3, 2))
    return pd.Series(g, index=v.index)


def _n_groups(sel: dict) -> tuple[str, int]:
    sort = sel.get("sort", "deciles")
    if sort == "top_bottom_percent":
        return sort, 3
    if sort in SORT_GROUPS:
        return sort, int(sel.get("n_groups") or SORT_GROUPS[sort])
    raise ValueError(f"unsupported sort {sort!r}")


def _grouper(sort: str, n: int, percent: Optional[float]):
    if sort == "top_bottom_percent":
        p = TOP_PERCENT_DEFAULT if percent is None else float(percent)
        return lambda v: assign_percent_groups(v, p)
    return lambda v: assign_groups(v, n)


def _op_params(spec: ExperimentSpec, name: str) -> dict:
    for o in spec.signal_ops:
        if o.op == name:
            return dict(o.params)
    return {}


# --------------------------------------------------------------------------- main
def build_targets(ctx, spec: ExperimentSpec) -> ConstructionResult:
    if spec.holding_months < 1:
        raise ValueError(f"holding_months must be >= 1, got {spec.holding_months}")
    if spec.weighting != "equal":
        raise NotImplementedError("only equal weighting is supported")

    dates, ex = formation_dates(ctx, spec)
    sel = dict(spec.selection or {})
    ds = sel.get("double_sort")

    elig = {d: ctx.eligible_mask(d, spec.min_history_days, spec.price_floor) for d in dates}
    eligible_counts = pd.Series({d: int(m.sum()) for d, m in elig.items()}, name="n_eligible", dtype="int64")
    eligible_counts.index = pd.DatetimeIndex(eligible_counts.index, name="date")

    rows, targets, skips, skipped = [], {}, [], []
    if ds:
        outer = registry.compute(ctx, dates, ds["outer_op"], _op_params(spec, ds["outer_op"]))
        inner = registry.compute(ctx, dates, ds["inner_op"], _op_params(spec, ds["inner_op"]))
        o_sort, o_n = _n_groups({"sort": ds.get("outer_sort", "quintiles"), "n_groups": ds.get("outer_n_groups")})
        i_sort, i_n = _n_groups({"sort": ds.get("inner_sort", "terciles"), "n_groups": ds.get("inner_n_groups")})
        o_group = _grouper(o_sort, o_n, ds.get("outer_percent"))
        i_group = _grouper(i_sort, i_n, ds.get("inner_percent"))
        o_fav = 1 if ds.get("outer_favored", "highest") == "lowest" else o_n
        fav = 1 if ds.get("inner_favored", "lowest") == "lowest" else i_n
        n_groups = i_n
    else:
        if not spec.signal_ops:
            raise ValueError("spec.signal_ops is empty")
        last = spec.signal_ops[-1]
        signal = registry.compute(ctx, dates, last.op, last.params)
        sort, n_groups = _n_groups(sel)
        group_fn = _grouper(sort, n_groups, sel.get("percent"))
        fav = 1 if sel.get("favored", "lowest") == "lowest" else n_groups

    group_targets: dict[int, dict] = {g: {} for g in range(1, n_groups + 1)}
    for d in dates:
        m = elig[d]
        exec_day = ex[d]
        if ds:
            ov = outer.loc[d][m].dropna()
            og = o_group(ov)
            top = og[og == o_fav].index
            iv = inner.loc[d].reindex(top).dropna()
            g = i_group(iv) if len(iv) else pd.Series(dtype="int64")
            df = pd.DataFrame({"ticker": ov.index, "outer_signal": ov.to_numpy(), "outer_group": og.to_numpy()})
            df["signal"] = inner.loc[d].reindex(ov.index).to_numpy()
            df["group"] = pd.array(g.reindex(ov.index).to_numpy(), dtype="Int64")
        else:
            v = signal.loc[d][m].dropna()
            g = group_fn(v)
            df = pd.DataFrame({"ticker": v.index, "signal": v.to_numpy(),
                               "group": pd.array(g.to_numpy(), dtype="Int64")})
        df.insert(0, "date", d)
        for k in range(1, n_groups + 1):
            names = np.sort(g.index[g.to_numpy() == k].to_numpy().astype(str))
            if len(names) < spec.min_names:
                skips.append({"formation_date": d, "execution_day": exec_day, "group": k,
                              "n_names": int(len(names)), "favored": k == fav,
                              "reason": f"group {k} has {len(names)} < min_names {spec.min_names}"})
                continue
            group_targets[k][exec_day] = pd.Series(1.0 / len(names), index=pd.Index(names, name="ticker"))
        selected = df["group"].eq(fav).fillna(False).to_numpy(dtype=bool)
        n_sel = int(selected.sum())
        ok = n_sel >= spec.min_names
        if ok:
            targets[exec_day] = group_targets[fav][exec_day]
        else:
            skipped.append(d)
        df["selected"] = selected & ok
        df["weight"] = np.where(df["selected"], 1.0 / max(n_sel, 1), 0.0)
        df["cohort"] = d
        rows.append(df)

    cols = ["date", "ticker", "signal", "group", "selected", "weight", "cohort"]
    signals_long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)
    extra = [c for c in signals_long.columns if c not in cols]
    signals_long = signals_long[cols + extra]
    signals_long["ticker"] = signals_long["ticker"].astype(str)
    K = int(spec.holding_months)
    cohort_log: list[dict] = []
    if K > 1:
        # targets / group_targets above are the one-month cohorts; net K of them per execution day
        tickers = ctx.tickers.astype(str)
        elig_fn = lambda d: tickers[elig[d]]  # noqa: E731

        def cohorts_of(per_exec: dict) -> dict:
            return {d: (per_exec[ex[d]].index if ex[d] in per_exec else None) for d in dates}

        targets, ev = net_cohort_targets(dates, ex, cohorts_of(targets), K, elig_fn, "strategy")
        cohort_log += ev
        for k in list(group_targets):
            group_targets[k], ev = net_cohort_targets(dates, ex, cohorts_of(group_targets[k]), K, elig_fn,
                                                      f"group_{k}")
            cohort_log += ev
    return ConstructionResult(signals_long=signals_long, targets=targets, group_targets=group_targets,
                              eligible_counts=eligible_counts, skipped_months=skipped, skip_log=skips,
                              n_groups=n_groups, favored_group=fav, formation_dates=dates, execution_days=ex,
                              holding_months=K, cohort_log=cohort_log)
