"""Stage 3: turn the paper's StrategySpec into the ExperimentSpec we test.

Explicit policies: map the signal to registry operations (rule table below),
equal weight, long-only favored group, our universe and dates, defaults for
anything unresolved. Every choice is logged as an AdaptationEntry with status
stated / assumed / adapted / unresolved.

Status is "blocked" when (a) can_implement is false and no check pass disputes
it, (b) a required input is missing from the snapshot, or (c) the signal maps
to no registry operation.

Counting: n_assumed counts assumed entries on MATERIAL_FIELDS (the rubric's
amber threshold is about material fields); n_adapted counts every adapted entry.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from research_engine.contracts import (
    EVAL_END, EVAL_START, MATERIAL_FIELDS, MIN_FORMATION_DAYS, MIN_HISTORY_DAYS, MIN_NAMES,
    MIN_NAMES_DOUBLE_SORT, N_GROUPS_DEFAULT, TOP_PERCENT_DEFAULT, WARMUP_START,
    AdaptationEntry, CheckPassResult, ExperimentSpec, SignalOp, StrategySpec,
)
from research_engine.extraction.evidence_check import normalize_text
from research_engine.validation.consistency import (
    check_consistency, group_direction, is_not_stated, parse_months, parse_window_lags,
)
from research_engine.validation.defaults import DEFAULTS, STRATEGY_FALLBACKS

_SORT_GROUPS = {"deciles": 10, "quintiles": 5, "terciles": 3}


# ----------------------------------------------------------------------------
# signal -> registry ops
# ----------------------------------------------------------------------------
def _months_or(spec_field_value: str, fallback: float) -> tuple[int, bool]:
    m = parse_months(spec_field_value)
    if m is None or m <= 0:
        return int(fallback), False
    return max(1, int(round(m))), True


def _ops_frog(spec):
    w = DEFAULTS["frog_window"]["value"]
    lags = parse_window_lags(f"{spec.formation_window.value} {spec.signal_formula_symbols.value}")
    months, skip = (lags[0] - lags[1] + 1, lags[1] - 1) if lags else (w["months"], w["skip"])
    # months t-12..t-2 is the paper's 12-month formation with the most recent month skipped;
    # the registered ops take (months=12, skip_months=1) and cover exactly that span.
    ops = [SignalOp(op="pret_12_2", params={"months": 12, "skip_months": 1}),
           SignalOp(op="info_discreteness", params={"months": 12, "skip_months": 1})]
    selection = {"sort": "quintiles", "n_groups": 5, "percent": None, "favored": "highest",
                 "double_sort": {"outer_op": "pret_12_2", "outer_sort": "quintiles", "outer_n_groups": 5,
                                 "outer_favored": "highest", "inner_op": "info_discreteness", "inner_sort": "terciles",
                                 "inner_n_groups": 3, "inner_favored": "lowest",
                                 "min_names": MIN_NAMES_DOUBLE_SORT}}
    return ops, selection, bool(lags)


def _ops_max(spec):
    m, ok = _months_or(spec.formation_window.value, 1)
    return [SignalOp(op="rolling_max_daily_return", params={"window_months": m, "min_days": MIN_FORMATION_DAYS})], None, ok


def _ops_high52(spec):
    return [SignalOp(op="price_to_rolling_high", params={"window_days": 252, "high_field": "high"})], None, True


def _ops_residual_vol(spec):
    m, ok = _months_or(spec.formation_window.value, 1)
    return [SignalOp(op="residual_vol", params={"window_months": m, "factors": "ff3", "min_days": MIN_FORMATION_DAYS})], None, ok


def _ops_oil(spec):
    m, ok = _months_or(spec.formation_window.value, STRATEGY_FALLBACKS["oil"]["formation_months"])
    return [SignalOp(op="oil_beta", params={"window_months": m})], None, ok


def _ops_seasonality(spec):
    """Single lag 12 by default; if the paper names several annual lags (e.g. 24, 36, 48, 60)
    average the lagged same-month returns over those lags (min 2 available)."""
    text = f"{spec.signal_formula_words.value} {spec.formation_window.value}"
    lags = sorted({int(x) for x in re.findall(r"\b(12|24|36|48|60|72|84|96|108|120)\b", text)})
    lags = [l for l in lags if l % 12 == 0]
    if len(lags) >= 2:
        return [SignalOp(op="avg_lagged_month_return", params={"lags": lags, "min_lags": 2})], None, True
    return [SignalOp(op="lagged_month_return", params={"lag_months": lags[0] if lags else 12})], None, True


def _ops_rolling_vol(spec):
    text = f"{spec.signal_formula_words.value} {spec.formation_window.value}".lower()
    weekly = bool(re.search(r"week", text))
    m, ok = _months_or(spec.formation_window.value, 36 if weekly else 1)
    return [SignalOp(op="rolling_vol", params={"window_months": m, "freq": "weekly" if weekly else "daily"})], None, ok


def _ops_momentum(spec):
    m, ok = _months_or(spec.formation_window.value, STRATEGY_FALLBACKS["momentum"]["formation_months"])
    return [SignalOp(op="cum_return", params={"months": m, "skip_months": _skip_months(spec)[0]})], None, ok


# order matters: most specific first
SIGNAL_RULES: list[tuple[str, re.Pattern, Callable]] = [
    ("frog", re.compile(r"information discreteness|frog[- ]in[- ]the[- ]pan|\bid\s*=|sgn\s*\(\s*pret", re.I), _ops_frog),
    ("high52", re.compile(r"52[- ]?week|price to (its |the )?(52|twelve|12)|nearness to (the )?(52|high)|"
                          r"ratio of (the )?current price to (the )?highest", re.I), _ops_high52),
    ("max", re.compile(r"\bmax\b|max(imum)? (daily )?return|largest daily return|highest daily return|"
                       r"extreme (positive )?(daily )?return", re.I), _ops_max),
    ("oil", re.compile(r"\boil\b|crude|\boilp\b", re.I), _ops_oil),
    ("lowvol", re.compile(r"idiosyncratic vol|idiosyncratic risk|residual|\bivol\b|fama[- ]french (three|3)", re.I),
     _ops_residual_vol),
    ("seasonality", re.compile(r"same (calendar )?month|seasonal|\blag(ged)?[ -]?(12|twelve)\b|"
                               r"(12|twelve)[- ]month lag|one year (earlier|ago|before|prior)|annual lag", re.I),
     _ops_seasonality),
    ("lowvol", re.compile(r"volatility|standard deviation|\bstd\b|variance", re.I), _ops_rolling_vol),
    ("momentum", re.compile(r"cumulative return|compound(ed)? return|momentum|past (\w+ )?(\d+|six|twelve|three|nine)"
                            r"[- ]month|prior (\d+|six|twelve)[- ]month|buy[- ]and[- ]hold return|past return|"
                            r"lagged return|formation[- ]period return", re.I), _ops_momentum),
]


def map_signal(spec: StrategySpec):
    """-> (slug, ops, selection_override, window_parsed) or None if unmapped."""
    text = normalize_text(f"{spec.signal_formula_words.value} {spec.signal_formula_symbols.value}")
    for slug, rx, fn in SIGNAL_RULES:
        if rx.search(text):
            ops, sel, ok = fn(spec)
            return slug, ops, sel, ok
    return None


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _skip_months(spec: StrategySpec) -> tuple[int, str, str]:
    """-> (skip months, status, reason)."""
    v = spec.skip_window.value
    if is_not_stated(v):
        return 0, "assumed", DEFAULTS["skip_months"]["text"]
    m = parse_months(v)
    if m is None:
        return 0, "assumed", f"could not parse '{v}'; {DEFAULTS['skip_months']['text']}"
    if 0 < m < 1:
        return 0, "adapted", "signals are monthly; a sub-month skip is dropped"
    return int(round(m)), "stated", "as stated"


def _price_floor(spec: StrategySpec) -> tuple[Optional[float], str, str]:
    u = spec.universe_and_filters.value or ""
    m = re.search(r"(exclud|drop|remov|screen|omit|delet|eliminat)\w*[^.]{0,80}\$\s*(\d+(?:\.\d+)?)", u, re.I)
    if not m:
        m = re.search(r"(\$)\s*(\d+(?:\.\d+)?)\s+price filter", u, re.I)
    if m:
        return None, "adapted", (f"paper excludes stocks priced below ${m.group(2)}; NOT applied: the snapshot's prices are "
                                 "split-adjusted, not as-traded, so a dollar floor would drop the wrong names (e.g. NVDA 2017-18)")
    return None, "assumed", DEFAULTS["price_floor"]["text"]


def _disputed(field_names: set[str], checks: list[CheckPassResult]) -> bool:
    return any(item.field in field_names for c in checks for item in c.items)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def adapt(spec: StrategySpec, run_id: str, paper_sha256: str, selected_experiment: Optional[str] = None, *,
          consistency_issues: Optional[list[dict]] = None,
          check_results: Optional[list[CheckPassResult]] = None,
          evidence: Optional[dict] = None) -> ExperimentSpec:
    issues = check_consistency(spec) if consistency_issues is None else consistency_issues
    checks = check_results or []
    log: list[AdaptationEntry] = []
    blocked: list[str] = []

    def entry(field, paper_value, our_value, status, reason):
        log.append(AdaptationEntry(field=field, paper_value=None if paper_value is None else str(paper_value),
                                   our_value=str(our_value), status=status, reason=reason))

    entry("selected_experiment", spec.selected_experiment, selected_experiment or spec.selected_experiment,
          "stated", "chosen by the user" if selected_experiment else "identified by the extraction")

    # --- blocking conditions -------------------------------------------------
    if not spec.can_implement:
        if _disputed({"can_implement", "cannot_implement_reasons"}, checks):
            entry("can_implement", "false", "evaluated", "adapted",
                  "check pass disputes the cannot-implement judgement; other blocking rules still apply")
        else:
            reasons = "; ".join(spec.cannot_implement_reasons) or "no reason given"
            blocked.append(f"paper cannot be implemented with our data: {reasons}")
    for iss in issues:
        if iss.get("tag") == "missing_input":
            blocked.append(f"required input not in snapshot: {iss['message']}")
        elif iss.get("tag") == "adaptable_input":
            entry("required_inputs", iss["message"].split("':")[0].strip("'"), "not used",
                  "adapted", iss["message"].split("': ", 1)[-1])

    # --- signal --------------------------------------------------------------
    mapped = map_signal(spec)
    if mapped is None:
        text = spec.signal_formula_words.value
        blocked.append(f"signal maps to no registry operation: {text}")
        slug, ops, sel_override, window_ok = "unmapped", [], None, False
    else:
        slug, ops, sel_override, window_ok = mapped
        entry("signal_formula_words", spec.signal_formula_words.value,
              "; ".join(f"{o.op}({', '.join(f'{k}={v}' for k, v in o.params.items())})" for o in ops),
              "stated", "mapped to registry operations by rule table")
    fb = STRATEGY_FALLBACKS.get(slug, {"favored": "highest", "formation_months": 1, "holding_months": 1})
    if mapped is not None:
        fw = spec.formation_window.value
        entry("formation_window", fw, ops[0].params.get("months", ops[0].params.get("window_months",
              ops[0].params.get("window_days", ops[0].params.get("lag_months")))),
              "stated" if window_ok and not is_not_stated(fw) else "assumed",
              "as stated" if window_ok and not is_not_stated(fw) else "window not stated or not parseable; strategy default")

    # --- selection -----------------------------------------------------------
    if sel_override is not None:
        selection = sel_override
        entry("sort_rule", spec.sort_rule, "quintiles on past return, then terciles on information discreteness",
              "adapted",
              "double sort per rule table; inner sort terciles with a 15-name floor")
        min_names = MIN_NAMES_DOUBLE_SORT
    else:
        min_names = MIN_NAMES
        sr = spec.sort_rule
        if sr in _SORT_GROUPS:
            selection = {"sort": sr, "n_groups": _SORT_GROUPS[sr], "percent": None}
            entry("sort_rule", sr, sr, "stated", "as stated")
        elif sr == "top_bottom_percent":
            pm = re.search(r"(\d+(?:\.\d+)?)\s*(%|percent)", f"{spec.breakpoints.value} {spec.favored_group.value}")
            pct = float(pm.group(1)) / 100 if pm else TOP_PERCENT_DEFAULT
            selection = {"sort": "top_bottom_percent", "n_groups": 3, "percent": pct}
            entry("sort_rule", sr, f"top/bottom {pct:.0%}", "stated" if pm else "assumed",
                  "as stated" if pm else DEFAULTS["top_percent"]["text"])
        else:
            selection = {"sort": "deciles", "n_groups": N_GROUPS_DEFAULT, "percent": None}
            entry("sort_rule", sr, "deciles", "adapted",
                  f"paper sort '{sr}' is not a ranked-group sort; {DEFAULTS['sort']['text']}")

        fav_dir = group_direction(spec.favored_group.value)
        if fav_dir is None:
            dis = group_direction(spec.disfavored_group.value)
            fav_dir = {"lowest": "highest", "highest": "lowest"}.get(dis)
        if fav_dir is None:
            selection["favored"] = fb["favored"]
            entry("favored_group", spec.favored_group.value, fb["favored"], "assumed",
                  "favored side not stated; strategy default, all groups are computed")
        elif spec.expected_sign == "unclear":
            selection["favored"] = fav_dir
            entry("favored_group", spec.favored_group.value, f"{fav_dir} group", "assumed",
                  "paper claims no unconditional relation (expected_sign unclear); the engine tests this group as a "
                  "default and reports every group")
        else:
            selection["favored"] = fav_dir
            entry("favored_group", spec.favored_group.value, f"{fav_dir} group", "stated", "as stated")
        selection["double_sort"] = None

    bp = spec.breakpoints.value
    if is_not_stated(bp):
        entry("breakpoints", bp, "equal-count over eligible names", "assumed", DEFAULTS["breakpoints"]["text"])
    elif re.search(r"nyse|size|market cap", bp, re.I):
        entry("breakpoints", bp, "equal-count over eligible names", "adapted",
              "no exchange or size breakpoints in our universe; equal-count sorts of eligible names")
    else:
        entry("breakpoints", bp, "equal-count over eligible names", "stated", "as stated")

    # --- weighting, long-only, sign ----------------------------------------
    entry("weighting", spec.weighting, "equal", "stated" if spec.weighting == "equal" else "adapted",
          "as stated" if spec.weighting == "equal" else "policy: equal weight everywhere")
    entry("disfavored_group", f"long {spec.favored_group.value} / short {spec.disfavored_group.value}",
          "long-only favored group; every group computed for the monotonicity table", "adapted",
          "policy: long-only, no short leg")
    if spec.expected_sign == "unclear":
        expected_sign = "favored_group_higher"
        entry("expected_sign", "unclear", expected_sign, "assumed", DEFAULTS["expected_sign"]["text"])
    else:
        expected_sign = spec.expected_sign
        entry("expected_sign", spec.expected_sign, expected_sign, "stated", "as stated")

    # --- holding, skip, rebalance -------------------------------------------
    hp = spec.holding_period.value
    h = parse_months(hp)
    if h is None:
        holding = DEFAULTS["holding_months"]["value"]
        entry("holding_period", hp, f"{holding} month", "assumed", DEFAULTS["holding_months"]["text"])
    elif h < 1:
        holding = 1
        entry("holding_period", hp, "1 month", "adapted", "sub-month holding adapted to the monthly cycle")
    else:
        holding = int(round(h))
        entry("holding_period", hp, f"{holding} months" if holding > 1 else "1 month", "stated",
              "as stated" + ("; overlapping monthly cohorts" if holding > 1 else ""))

    if sel_override is None:
        skip, s_status, s_reason = _skip_months(spec)
    else:
        skip, s_status, s_reason = ops[0].params.get("skip_months", 0), "stated", "skip embedded in the t-12..t-2 window"
    entry("skip_window", spec.skip_window.value, f"{skip} months", s_status, s_reason)

    rb = spec.rebalance_frequency.value
    rbm = parse_months(rb)
    if is_not_stated(rb) or rbm is None:
        entry("rebalance_frequency", rb, "monthly", "assumed", DEFAULTS["rebalance"]["text"])
    elif abs(rbm - 1) < 1e-9:
        entry("rebalance_frequency", rb, "monthly", "stated", "as stated (cohorts rebalance monthly)")
    else:
        entry("rebalance_frequency", rb, "monthly", "adapted", "engine rebalances monthly")

    # --- universe, filters, dates -------------------------------------------
    floor, f_status, f_reason = _price_floor(spec)
    entry("universe_and_filters", spec.universe_and_filters.value,
          ExperimentSpec.model_fields["universe"].default, "adapted",
          "policy: our 415-name S&P 500 survivor universe")
    entry("price_floor", None if f_status == "assumed" else spec.universe_and_filters.value,
          "none" if floor is None else f"${floor:g}", f_status, f_reason)
    entry("sample_period", spec.headline_result.value, f"{EVAL_START}..{EVAL_END} (warm-up from {WARMUP_START})",
          "adapted", "policy: our snapshot sample")
    if slug == "frog":
        entry("inner_sort", "quintiles within the past-return quintile (paper)", "terciles", "adapted",
              "415-name universe: quintile-within-quintile leaves ~17 names, under the 20-name floor; terciles with a 15-name floor")
    if slug == "oil":
        entry("timing_signal", "conditional timing signal with threshold h selects the +/0/- oil-exposure portfolio (optimizer-built)",
              "dropped: unconditional monthly sort on each stock's trailing oil beta, long the favored group", "adapted",
              "the note's signal is built from realized future oil returns (an oracle); the engine tests the exposure sort only")
        entry("oil_series", "monthly oil price level (paper's input)", "data/oil_prices.csv, ends 2024-03", "adapted",
              "after the oil series ends, regressions use available paired observations until fewer than 24 remain; "
              "a skipped formation issues no new target, so existing holdings carry forward")
    if any(o.op == "avg_lagged_month_return" for o in ops):
        entry("min_lags", "all listed annual lags", "mean of available lags, at least 2", "assumed",
              "snapshot starts 2012-01, so 2015-16 formations have only lags 24 and 36 of the paper's 24-60")

    window_days = 0
    for o in ops:
        p = o.params
        if "window_days" in p:
            window_days = max(window_days, int(p["window_days"]))
        months = p.get("months", p.get("window_months", p.get("lag_months", 0))) + p.get("skip", 0)
        window_days = max(window_days, int(round(months * 21)))
    min_hist = max(MIN_HISTORY_DAYS, window_days)
    entry("min_history_days", None, min_hist, "assumed", DEFAULTS["min_history_days"]["text"])
    entry("min_formation_days", None, MIN_FORMATION_DAYS, "assumed", DEFAULTS["min_formation_days"]["text"])

    for d in spec.unresolved_details:
        entry("unresolved_detail", d, "engine default or convention", "unresolved",
              "left open by the paper; see defaults")

    if evidence:
        bad = [f for f, r in evidence.items() if not r.get("valid")]
        if bad:
            entry("evidence", ", ".join(bad), "reported", "unresolved", "quotes not found on the cited page (or adjacent)")

    n_assumed = sum(1 for a in log if a.status == "assumed" and a.field in MATERIAL_FIELDS)
    n_adapted = sum(1 for a in log if a.status == "adapted")

    return ExperimentSpec(
        run_id=run_id, paper_sha256=paper_sha256, strategy_slug=slug,
        status="blocked" if blocked else "ready", blocked_reasons=blocked,
        signal_ops=ops, selection=selection, weighting="equal",
        holding_months=holding, skip_months=int(skip), min_names=min_names,
        price_floor=floor, min_history_days=min_hist, min_formation_days=MIN_FORMATION_DAYS,
        expected_sign=expected_sign, adaptations=log, n_assumed=n_assumed, n_adapted=n_adapted,
    )
