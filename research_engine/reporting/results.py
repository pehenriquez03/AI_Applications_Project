"""Assemble results.json: everything the dossier shows and the narrative may cite.

`assemble_results(run_dir)` reads the run folder and returns a JSON-safe dict.
Every input file is optional: a missing or unreadable file yields a null
section, never an exception. Floats are rounded to 6 significant digits so the
narrative can quote them and the number check can find them.

Top-level keys (CONTRACTS.md): paper, extraction, adaptation, data,
construction, performance, robustness, economics, interpretation, limitations,
llm_usage, plus run_id, status, status_reasons, files. `rubric`,
`recommendation` and `recommendation_reasons` are added by the pipeline after
`reporting.rubric.recommend`.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

from research_engine import contracts as C
from research_engine.contracts import RunFiles

log = logging.getLogger("engine.reporting")

PRICE_IN_PER_MTOK = 2.0
PRICE_OUT_PER_MTOK = 10.0
CRITICAL_FIELDS = ("signal_formula_words", "signal_formula_symbols", "formation_window", "skip_window",
                   "holding_period", "sort_rule", "favored_group", "expected_sign")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
    except Exception as e:  # noqa: BLE001
        log.warning("could not read %s: %s", path, e)
        return None


def _read_parquet(path: Path):
    try:
        import pandas as pd
        if not Path(path).exists():
            return None
        return pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001
        log.warning("could not read %s: %s", path, e)
        return None


def clean(obj: Any, sig: int = 6) -> Any:
    """JSON-safe copy: numpy scalars -> python, NaN/inf -> None, floats to `sig` significant digits."""
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            obj = obj.item()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        if obj == 0:
            return 0.0
        return float(f"{obj:.{sig}g}")
    if isinstance(obj, dict):
        return {str(k): clean(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v, sig) for v in obj]
    return str(obj)


def dig(d: Any, *paths: str, default=None):
    """First non-None value among dotted paths, e.g. dig(sc, "performance.strategy_net.cagr", "cagr")."""
    for p in paths:
        cur = d
        ok = True
        for part in p.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return default


def _ann_stats(r) -> dict:
    """CAGR, vol, Sharpe (zero rf), max drawdown from a daily simple-return Series."""
    import numpy as np
    r = r.dropna()
    if len(r) < 2:
        return {}
    growth = float((1 + r).prod())
    cagr = growth ** (252 / len(r)) - 1
    vol = float(r.std() * np.sqrt(252))
    w = (1 + r).cumprod()
    return {"cagr": cagr, "ann_vol": vol, "sharpe": (cagr / vol) if vol else None,
            "max_drawdown": float((w / w.cummax() - 1).min()), "cum_return": growth - 1}


# ----------------------------------------------------------------------------
# sections
# ----------------------------------------------------------------------------
def _paper(run_dir: Path, meta: Optional[dict], cons: Optional[dict]) -> Optional[dict]:
    if meta is None and cons is None:
        return None
    spec = (cons or {}).get("spec") or {}
    meta = meta or {}
    return {"title": meta.get("title") or spec.get("paper_title"), "authors": spec.get("authors"),
            "year": spec.get("year"), "sha256": meta.get("sha256"), "source_path": meta.get("source_path"),
            "n_pages": meta.get("n_pages"),
            "selected_experiment": spec.get("selected_experiment") or meta.get("selected_experiment")}


def _field_value(spec: dict, f: str):
    v = spec.get(f)
    if isinstance(v, dict):
        return v.get("value"), v.get("evidence") or {}
    return v, spec.get(f + "_evidence") or {}


def _extraction(cons: Optional[dict], validation: Optional[dict]) -> Optional[dict]:
    if cons is None:
        return None
    spec = cons.get("spec") or {}
    agreement = cons.get("agreement") or {}
    evidence = (validation or {}).get("evidence") or {}
    fields = {}
    for f in C.MATERIAL_FIELDS:
        val, ev = _field_value(spec, f)
        e = evidence.get(f) or {}
        fields[f] = {"value": val, "page": e.get("page", ev.get("page")), "quote": ev.get("quote"),
                     "evidence_valid": e.get("valid") if e else None, "agreement": agreement.get(f)}
    valid = [v["evidence_valid"] for v in fields.values() if v["evidence_valid"] is not None]
    checks = (validation or {}).get("check_pass") or []
    items = [it for c in checks for it in (c.get("items") or [])]
    critical = [it for it in items if it.get("field") in CRITICAL_FIELDS]
    agr = [v for v in agreement.values() if isinstance(v, (int, float))]
    return {
        "n_runs": cons.get("n_runs"),
        "can_implement": spec.get("can_implement"),
        "cannot_implement_reasons": spec.get("cannot_implement_reasons") or [],
        "unresolved_details": spec.get("unresolved_details") or [],
        "fields": fields,
        "mean_field_agreement": (sum(agr) / len(agr)) if agr else None,
        "evidence_valid_count": sum(bool(v) for v in valid) if valid else None,
        "evidence_checked_count": len(valid) if valid else None,
        "evidence_valid_rate": (sum(bool(v) for v in valid) / len(valid)) if valid else None,
        "consistency_issues": (validation or {}).get("consistency_issues") if validation else None,
        "check_pass_disagreements": items if validation else None,
        "n_check_disagreements": (validation or {}).get("n_check_disagreements", len(items)) if validation else None,
        "n_critical_check_disagreements": len(critical) if validation else None,
        "critical_fields": list(CRITICAL_FIELDS),
    }


def _adaptation(log_: Optional[list], exp: Optional[dict]) -> Optional[dict]:
    if log_ is None and exp is None:
        return None
    entries = log_ if log_ is not None else (exp or {}).get("adaptations") or []
    n_assumed = sum(e.get("status") == "assumed" for e in entries)
    n_adapted = sum(e.get("status") == "adapted" for e in entries)
    if not entries and exp:
        n_assumed, n_adapted = exp.get("n_assumed", 0), exp.get("n_adapted", 0)
    out = {"log": entries, "n_assumed": n_assumed, "n_adapted": n_adapted,
           "n_stated": sum(e.get("status") == "stated" for e in entries),
           "n_unresolved": sum(e.get("status") == "unresolved" for e in entries)}
    if exp:
        out.update({k: exp.get(k) for k in ("strategy_slug", "status", "blocked_reasons", "selection",
                                            "weighting", "holding_months", "skip_months", "min_names",
                                            "price_floor", "min_history_days", "min_formation_days",
                                            "expected_sign", "cost_bps", "execution")})
        out["signal_ops"] = exp.get("signal_ops")
    return out


def _data(dc: Optional[dict], exp: Optional[dict]) -> Optional[dict]:
    if dc is None and exp is None:
        return None
    dc = dc or {}
    exp = exp or {}
    excluded = dc.get("excluded") or []
    return {"snapshot": dc.get("snapshot", exp.get("snapshot")), "n_universe": dc.get("n_universe"),
            "n_excluded": len(excluded) if isinstance(excluded, list) else excluded,
            "excluded": excluded, "eligible_by_year": dc.get("eligible_by_year"),
            "survivor_universe": dc.get("survivor_universe"), "calendar_days": dc.get("calendar_days"),
            "universe": exp.get("universe"), "eval_start": exp.get("eval_start", C.EVAL_START),
            "eval_end": exp.get("eval_end", C.EVAL_END), "warmup_start": exp.get("warmup_start", C.WARMUP_START),
            "benchmarks": list(exp.get("benchmarks") or C.BENCHMARKS)}


def _construction(run_dir: Path, plan: Optional[dict], exp: Optional[dict]) -> Optional[dict]:
    sig = _read_parquet(run_dir / RunFiles.signals)
    if sig is None and plan is None:
        return None
    out: dict = {"skipped_months": (plan or {}).get("skipped_months") or [],
                 "holding_months": (plan or {}).get("holding_months")}
    out["n_skipped_months"] = len(out["skipped_months"])
    if sig is not None and len(sig):
        sel = sig[sig["selected"].astype(bool)] if "selected" in sig else sig.iloc[0:0]
        per = sel.groupby("date").size() if len(sel) else None
        out.update({"n_months": int(sig["date"].nunique()),
                    "first_month": str(sig["date"].min())[:10], "last_month": str(sig["date"].max())[:10],
                    "mean_names": float(per.mean()) if per is not None else None,
                    "min_names": int(per.min()) if per is not None else None,
                    "max_names": int(per.max()) if per is not None else None,
                    "mean_eligible": float(sig.groupby("date").size().mean())})
        if per is not None:
            last = sel["date"].max()
            favored = ((exp or {}).get("selection") or {}).get("favored", "highest")
            m = sel[sel["date"] == last].sort_values("signal", ascending=(favored == "lowest")).head(10)
            out["sample_month"] = str(last)[:10]
            out["sample_top10"] = [{"ticker": r.ticker, "signal": float(r.signal),
                                    "group": int(r.group) if "group" in m and r.group == r.group else None,
                                    "weight": float(r.weight) if "weight" in m else None}
                                   for r in m.itertuples()]
    trades = _read_parquet(run_dir / RunFiles.trades)
    if trades is not None and len(trades):
        rebalances = trades[trades["reason"] == "rebalance"] if "reason" in trades else trades
        if len(rebalances):
            out["last_rebalance"] = str(rebalances["date"].max())[:10]
    if exp and exp.get("strategy_slug") == "oil":
        out["coverage_note"] = (
            f"The oil input ends in March 2024. The regression uses remaining paired observations; "
            f"the last selected formation is {out.get('sample_month', 'unavailable')} and the last rebalance "
            f"is {out.get('last_rebalance', 'unavailable')}. Missing formations issue no new target: "
            "existing holdings carry forward and their subsequent returns remain in the evaluation."
        )
    return out


def _performance(run_dir: Path, sc: Optional[dict]) -> Optional[dict]:
    ret = _read_parquet(run_dir / RunFiles.returns)
    pv = _read_parquet(run_dir / RunFiles.portfolio_value)
    if sc is None and ret is None:
        return None
    sc = sc or {}
    perf = dict(sc.get("performance") or {})
    source = "scorecard_strategy"
    if ret is not None:
        for col in ("strategy_net", "strategy_gross", "spy", "ew_universe"):
            if col in ret and not perf.get(col):
                perf[col] = _ann_stats(ret[col])
                source = "returns.parquet (computed here; scorecard missing)"
    get = lambda col, k: dig(perf, f"{col}.{k}", f"{col}.annualized_return" if k == "cagr" else f"{col}.{k}")
    headline = {
        "strategy_net_cagr": get("strategy_net", "cagr") if perf.get("strategy_net") else dig(sc, "annualized_return_net"),
        "strategy_gross_cagr": get("strategy_gross", "cagr"),
        "ew_universe_cagr": get("ew_universe", "cagr"),
        "spy_cagr": get("spy", "cagr"),
        "strategy_net_sharpe": get("strategy_net", "sharpe"),
        "ew_universe_sharpe": get("ew_universe", "sharpe"),
        "spy_sharpe": get("spy", "sharpe"),
        "strategy_net_max_drawdown": get("strategy_net", "max_drawdown"),
        "alpha_ann": dig(sc, "alpha.ff4_alpha_ann", "alpha.alpha_ann", "alpha.alpha_annual", "alpha_ann"),
        "alpha_t": dig(sc, "alpha.alpha_t", "alpha_t"),
        "capm_beta": dig(sc, "alpha.capm_beta", "capm_beta"),
        "spanning_alpha_t": dig(sc, "spanning.alpha_t"),
    }
    c, e = headline["strategy_net_cagr"], headline["ew_universe_cagr"]
    headline["excess_cagr_vs_ew"] = (c - e) if c is not None and e is not None else None
    s = headline["spy_cagr"]
    headline["excess_cagr_vs_spy"] = (c - s) if c is not None and s is not None else None
    out = {"headline": headline, "by_series": perf, "source": source,
           "relative": sc.get("relative"), "alpha": sc.get("alpha"), "spanning": sc.get("spanning"),
           "turnover": sc.get("turnover"), "costs": sc.get("costs"), "holdings": sc.get("holdings"),
           "capacity": sc.get("capacity")}
    if pv is not None and "n_holdings" in pv:
        out["mean_holdings_daily"] = float(pv["n_holdings"].mean())
    if ret is not None:
        out["n_days"] = int(len(ret))
        out["first_day"] = str(ret.index.min())[:10]
        out["last_day"] = str(ret.index.max())[:10]
    return out


def _cost_key(grid: dict, bps: int):
    for k in (str(bps), bps, f"{bps}bps", f"{bps}_bps"):
        if k in grid:
            return grid[k]
    return None


def _robustness(sc: Optional[dict]) -> Optional[dict]:
    rob = (sc or {}).get("robustness")
    if not rob:
        return None
    grid = rob.get("cost_grid") or {}
    subs = rob.get("subperiods") or {}
    groups = rob.get("groups") or {}

    def holds(entry, benchmark_cagr=None) -> Optional[bool]:
        if not isinstance(entry, dict):
            return None
        t = dig(entry, "alpha_t")
        x = dig(entry, "excess_cagr_vs_ew")
        if x is None and dig(entry, "cagr") is not None and dig(entry, "ew_cagr") is not None:
            x = entry["cagr"] - entry["ew_cagr"]
        if x is None and entry.get("cagr") is not None and benchmark_cagr is not None:
            x = entry["cagr"] - benchmark_cagr
        if t is None or x is None:
            return None
        return bool(t > 0 and x > 0)

    return {"cost_grid": grid, "subperiods": subs, "groups": groups,
            "holds_at_bps": {str(b): holds(_cost_key(grid, b), dig(sc, "performance.ew_universe.cagr")) for b in C.COST_BPS_GRID},
            "holds_in_subperiod": {k: holds(v) for k, v in subs.items()},
            "rho": groups.get("rho"), "rho_in_expected_direction": groups.get("rho_in_expected_direction"),
            "holds_definition": "alpha t-statistic > 0 and net CAGR above the EW universe"}


def _economics(econ: Optional[dict], exp: Optional[dict]) -> Optional[dict]:
    if econ is None:
        return None
    q = econ.get("quantiles") or {}
    out = dict(econ)
    out["headline"] = {
        "benchmark": econ.get("benchmark", C.ECON_BENCHMARK),
        "incremental_fees_pv_quantiles": q.get("incremental_fees_pv"),
        "expected_incremental_fees_pv": econ.get("expected_incremental_fees_pv"),
        "frac_npv_positive": econ.get("frac_npv_positive"),
        "frac_npv_positive_demeaned": econ.get("frac_npv_positive_demeaned"),
        "npv_quantiles": q.get("npv"),
    }
    out["fund"] = (exp or {}).get("fund") or dict(C.FUND)
    out["research_cost_inputs"] = (exp or {}).get("research_cost") or dict(C.RESEARCH_COST)
    out["monte_carlo_inputs"] = (exp or {}).get("monte_carlo") or dict(C.MONTE_CARLO)
    return out


def _llm_usage(rows: Optional[list]) -> Optional[dict]:
    if rows is None:
        return None
    rows = [r for r in rows if not str(r.get("call", "")).startswith("narrative")]  # exclude the self-referential narrative call
    tin = sum(int(r.get("input_tokens", 0) or 0) for r in rows)
    tout = sum(int(r.get("output_tokens", 0) or 0) for r in rows)
    fresh = [r for r in rows if not r.get("cached")]
    fin = sum(int(r.get("input_tokens", 0) or 0) for r in fresh)
    fout = sum(int(r.get("output_tokens", 0) or 0) for r in fresh)
    return {"n_calls": len(rows), "n_cached": len(rows) - len(fresh),
            "input_tokens": tin, "output_tokens": tout,
            "cost_usd": tin * PRICE_IN_PER_MTOK / 1e6 + tout * PRICE_OUT_PER_MTOK / 1e6,
            "cost_usd_uncached_only": fin * PRICE_IN_PER_MTOK / 1e6 + fout * PRICE_OUT_PER_MTOK / 1e6,
            "price_per_mtok": {"input": PRICE_IN_PER_MTOK, "output": PRICE_OUT_PER_MTOK},
            "note": "cost_usd counts every call at list price (as charged to research cost), cached calls included",
            "calls": rows}


def _limitations(res: dict, exp: Optional[dict]) -> list[str]:
    lim = []
    data = res.get("data") or {}
    n_ex = data.get("n_excluded")
    n_u = data.get("n_universe")
    lim.append("Survivor universe: today's index constituents only"
               + (f" ({n_u} names; {n_ex} excluded for missing Yahoo data)" if n_u is not None else
                  (f" ({n_ex} excluded for missing Yahoo data)" if n_ex is not None else ""))
               + "; delisted names are absent, which flatters returns.")
    ad = res.get("adaptation") or {}
    log_ = ad.get("log") or []
    w = next((e for e in log_ if e.get("field") == "weighting"), None)
    lim.append("Equal weighting" + (f" (paper: {w.get('paper_value')})" if w and w.get("paper_value") else "") + ".")
    lim.append("Long-only adaptation: the strategy holds the favored group only; no short leg.")
    h = next((e for e in log_ if e.get("field") in ("holding_period", "holding_months")
              and e.get("status") == "adapted"), None)
    if h:
        lim.append(f"Holding period adapted: paper {h.get('paper_value')}, tested {h.get('our_value')} ({h.get('reason')}).")
    lim.append(f"Sample {str(data.get('eval_start', C.EVAL_START))[:4]}-{str(data.get('eval_end', C.EVAL_END))[:4]} "
               "lies entirely after publication.")
    lim.append("Cash earns zero yield.")
    lim.append("Returns use Yahoo adjusted-close conventions (dividends and splits folded in).")
    lim.append("Effort reduction from the engine is not measured.")
    if ad:
        lim.append(f"{ad.get('n_assumed', 0)} fields assumed (paper silent; labeled defaults) and "
                   f"{ad.get('n_adapted', 0)} fields adapted.")
    econ = res.get("economics") or {}
    if econ:
        lim.append("The 2/20 fee schedule with a hard hurdle is an assumption for a long-only fund; "
                   "incremental fees are measured against the EW universe fund.")
    coverage = (res.get("construction") or {}).get("coverage_note")
    if coverage:
        lim.append(coverage)
    interp = res.get("interpretation") or {}
    if interp.get("preliminary"):
        lim.append("Interpretation accuracy is scored against a preliminary reference.")
    return lim


def _status(run_dir: Path, exp: Optional[dict], sc: Optional[dict], ret_exists: bool) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if exp and exp.get("status") == "blocked":
        return "blocked", list(exp.get("blocked_reasons") or ["blocked by validation"])
    prog = _read_json(run_dir / RunFiles.progress)
    if prog:
        for st in prog.get("stages", []):
            if st.get("status") == "blocked" and st.get("n", 0) < 9:
                return "blocked", [st.get("detail") or f"stage {st.get('n')} blocked"]
    if exp is None:
        reasons.append("experiment_spec.json missing")
    if sc is None:
        reasons.append("scorecard_strategy.json missing")
    if not ret_exists:
        reasons.append("returns.parquet missing")
    if exp is None and not ret_exists:
        return "blocked", reasons
    return ("incomplete" if reasons else "complete"), reasons


# ----------------------------------------------------------------------------
def assemble_results(run_dir) -> dict:
    run_dir = Path(run_dir)
    meta = _read_json(run_dir / RunFiles.paper_meta)
    cons = _read_json(run_dir / RunFiles.spec_consensus)
    validation = _read_json(run_dir / "validation.json")
    exp = _read_json(run_dir / RunFiles.experiment_spec)
    alog = _read_json(run_dir / RunFiles.adaptation_log)
    dc = _read_json(run_dir / "data_check.json")
    plan = _read_json(run_dir / RunFiles.plan)
    sc = _read_json(run_dir / RunFiles.scorecard_strategy)
    si = _read_json(run_dir / RunFiles.scorecard_interpretation)
    econ = _read_json(run_dir / RunFiles.economics)
    usage = _read_json(run_dir / RunFiles.usage)

    def safe(fn, *a):
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001  never crash the dossier
            log.warning("results section %s failed: %s", fn.__name__, e)
            return None

    status, status_reasons = _status(run_dir, exp, sc, (run_dir / RunFiles.returns).exists())
    res: dict = {
        "run_id": run_dir.name,
        "status": status,
        "status_reasons": status_reasons,
        "strategy_slug": (exp or {}).get("strategy_slug"),
        "paper": safe(_paper, run_dir, meta, cons),
        "extraction": safe(_extraction, cons, validation),
        "adaptation": safe(_adaptation, alog, exp),
        "data": safe(_data, dc, exp),
        "construction": safe(_construction, run_dir, plan, exp),
        "performance": safe(_performance, run_dir, sc),
        "robustness": safe(_robustness, sc),
        "economics": safe(_economics, econ, exp),
        "interpretation": si,
        "llm_usage": safe(_llm_usage, usage),
    }
    res["limitations"] = safe(_limitations, res, exp) or []
    res["files"] = sorted(p.name for p in run_dir.iterdir() if p.is_file()) if run_dir.exists() else []
    return clean(res)
