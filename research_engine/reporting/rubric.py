"""Declared rules that turn results.json into one recommendation.

The model never decides. These rules are copied into results["rubric"]["rules"].
"""
from __future__ import annotations

from typing import Optional

from .results import dig

RULES = """\
Flags (each computed from results.json):
1. implementation_reliability (extraction checks only, not independently verified accuracy): green if n_assumed <= 3 AND evidence_valid_rate >= 0.8 AND no critical check-pass disagreements (check-pass items on signal_formula_words, signal_formula_symbols, formation_window, skip_window, holding_period, sort_rule, favored_group, expected_sign); amber if not green and n_assumed <= 6; red otherwise.
2. historical_performance: positive if strategy net CAGR > EW universe CAGR AND strategy net Sharpe > EW universe Sharpe; mixed if exactly one holds; negative if neither; unknown if either input is missing.
3. robustness: a test 'holds' when the alpha t-statistic is > 0 (sign only; the t-statistic is reported, not a gate) AND net CAGR exceeds the EW universe CAGR. strong if it holds at 25 bps AND in both sub-periods AND the ranked-group Spearman rho is in the expected direction; moderate if it holds at 10 bps AND in at least one sub-period; unknown when required inputs are missing; weak otherwise.
4. economics: from the fraction of Monte Carlo paths with incremental-fee NPV (vs the EW universe fund, less research cost) > 0: positive if >= 0.6; uncertain if >= 0.4 and < 0.6; negative if < 0.4; unknown if missing.
5. limitations: count = n_assumed + n_adapted (reported, not a gate).
Recommendation (first match wins):
a. 'unable to evaluate' if results status is blocked or incomplete.
b. 'pursue' if performance positive AND robustness strong AND economics positive AND reliability green.
c. 'deprioritize' if performance negative OR (robustness weak AND economics negative).
d. 'investigate further' otherwise.
"""

RULE_PARAMS = {"reliability_green_max_assumed": 3, "reliability_green_min_evidence_valid_rate": 0.8,
               "reliability_amber_max_assumed": 6, "econ_positive_min_frac": 0.6, "econ_uncertain_min_frac": 0.4,
               "robust_strong_cost_bps": 25, "robust_moderate_cost_bps": 10}


def _pct(x, d=1) -> str:
    return f"{x * 100:.{d}f}%"


def _money(x) -> str:
    if x is None:
        return "n/a"
    a = abs(x)
    sign = "-" if x < 0 else ""
    if a >= 1e6:
        return f"{sign}${a / 1e6:.1f}M"
    if a >= 1e3:
        return f"{sign}${a / 1e3:.0f}k"
    return f"{sign}${a:.0f}"


def _reliability(res: dict) -> tuple[str, dict]:
    ad = res.get("adaptation") or {}
    ex = res.get("extraction") or {}
    interp = res.get("interpretation") or {}
    n_assumed = ad.get("n_assumed")
    evr = interp.get("evidence_valid_rate")
    if evr is None:
        evr = ex.get("evidence_valid_rate")
    n_crit = ex.get("n_critical_check_disagreements") or 0
    inputs = {"n_assumed": n_assumed, "evidence_valid_rate": evr, "n_critical_check_disagreements": n_crit}
    if n_assumed is None:
        return "red", inputs
    if n_assumed <= 3 and evr is not None and evr >= 0.8 and n_crit == 0:
        return "green", inputs
    if n_assumed <= 6:
        return "amber", inputs
    return "red", inputs


def _performance(res: dict) -> tuple[str, dict]:
    h = dig(res, "performance.headline") or {}
    c, e = h.get("strategy_net_cagr"), h.get("ew_universe_cagr")
    s, se = h.get("strategy_net_sharpe"), h.get("ew_universe_sharpe")
    inputs = {"strategy_net_cagr": c, "ew_universe_cagr": e, "strategy_net_sharpe": s, "ew_universe_sharpe": se}
    if None in (c, e, s, se):
        return "unknown", inputs
    n = int(c > e) + int(s > se)
    return {2: "positive", 1: "mixed", 0: "negative"}[n], inputs


def _robustness(res: dict) -> tuple[str, dict]:
    r = res.get("robustness") or {}
    hb = r.get("holds_at_bps") or {}
    hs = r.get("holds_in_subperiod") or {}
    rho_ok = r.get("rho_in_expected_direction") is True
    sub_vals = list(hs.values())
    inputs = {"holds_at_25bps": hb.get("25"), "holds_at_10bps": hb.get("10"), "holds_in_subperiod": hs,
              "rho": r.get("rho"), "rho_in_expected_direction": r.get("rho_in_expected_direction")}
    if hb.get("25") is True and len(sub_vals) >= 2 and all(v is True for v in sub_vals) and rho_ok:
        return "strong", inputs
    if hb.get("10") is True and any(v is True for v in sub_vals):
        return "moderate", inputs
    if (hb.get("10") is None or hb.get("25") is None or len(sub_vals) < 2
            or any(v is None for v in sub_vals) or r.get("rho_in_expected_direction") is None):
        return "unknown", inputs
    return "weak", inputs


def _economics(res: dict) -> tuple[str, dict]:
    f = dig(res, "economics.frac_npv_positive")
    inputs = {"frac_npv_positive": f}
    if f is None:
        return "unknown", inputs
    if f >= 0.6:
        return "positive", inputs
    if f >= 0.4:
        return "uncertain", inputs
    return "negative", inputs


def _reasons(res: dict, flags: dict, inputs: dict, rec: str) -> list[str]:
    out = []
    p = inputs["historical_performance"]
    if flags["historical_performance"] != "unknown":
        out.append(f"Historical performance is {flags['historical_performance']}: net CAGR {_pct(p['strategy_net_cagr'], 2)} "
                   f"vs {_pct(p['ew_universe_cagr'], 2)} for the EW universe; Sharpe {p['strategy_net_sharpe']:.2f} "
                   f"vs {p['ew_universe_sharpe']:.2f}.")
    else:
        out.append("Historical performance could not be scored (scorecard inputs missing).")
    rb = inputs["robustness"]
    h = dig(res, "performance.headline") or {}
    t = h.get("alpha_t")
    parts = []
    for bps in (10, 25):
        value = rb[f"holds_at_{bps}bps"]
        label = "holds" if value is True else "fails" if value is False else "not evaluated (inputs missing)"
        parts.append(f"{bps} bps: {label}")
    hs = rb["holds_in_subperiod"] or {}
    evaluated = [v for v in hs.values() if v is not None]
    parts.append(f"{sum(v is True for v in evaluated)} of {len(evaluated)} evaluated sub-periods hold; "
                 f"{len(hs) - len(evaluated)} not evaluated")
    if rb["rho"] is not None:
        direction = rb["rho_in_expected_direction"]
        label = "in" if direction is True else "not in" if direction is False else "not evaluated against"
        parts.append(f"group rho {rb['rho']:.2f} {label} the expected direction")
    out.append(f"Robustness is {flags['robustness']}: " + ", ".join(parts)
               + (f"; alpha t {t:.2f} (reported, not a gate)." if t is not None else "."))
    e = dig(res, "economics") or {}
    if flags["economics"] != "unknown":
        med = dig(e, "quantiles.incremental_fees_pv.p50")
        out.append(f"Economics are {flags['economics']}: P(NPV > 0) {_pct(e['frac_npv_positive'], 0)}"
                   + (f", median incremental fee PV {_money(med)} vs the EW universe fund." if med is not None else "."))
    else:
        out.append("Economics could not be scored (economics.json missing).")
    ri = inputs["implementation_reliability"]
    evr = ri["evidence_valid_rate"]
    out.append(f"Extraction checks are {flags['implementation_reliability']}: {ri['n_assumed']} fields assumed, "
               f"{flags['limitations_count'] - (ri['n_assumed'] or 0)} adapted"
               + (f", evidence valid for {_pct(evr, 0)} of material fields" if evr is not None else "")
               + f", {ri['n_critical_check_disagreements']} critical check disagreements. This flag does not establish independent interpretation accuracy.")
    return out


def recommend(results: dict) -> dict:
    status = results.get("status")
    rel, rel_in = _reliability(results)
    perf, perf_in = _performance(results)
    rob, rob_in = _robustness(results)
    eco, eco_in = _economics(results)
    ad = results.get("adaptation") or {}
    lim = (ad.get("n_assumed") or 0) + (ad.get("n_adapted") or 0)
    flags = {"implementation_reliability": rel, "historical_performance": perf, "robustness": rob,
             "economics": eco, "limitations_count": lim}
    inputs = {"implementation_reliability": rel_in, "historical_performance": perf_in,
              "robustness": rob_in, "economics": eco_in}
    if status in ("blocked", "incomplete"):
        rec = "unable to evaluate"
        why = list(results.get("status_reasons") or []) or [f"run {status}"]
        reasons = [f"Unable to evaluate: {'; '.join(str(w) for w in why)}."]
    else:
        if perf == "positive" and rob == "strong" and eco == "positive" and rel == "green":
            rec = "pursue"
        elif perf == "negative" or (rob == "weak" and eco == "negative"):
            rec = "deprioritize"
        else:
            rec = "investigate further"
        reasons = _reasons(results, flags, inputs, rec)
    return {"recommendation": rec, "reasons": reasons,
            "rubric": {"flags": flags, "inputs": inputs, "rules": RULES, "params": RULE_PARAMS}}


__doc__ = (__doc__ or "") + "\n" + RULES
