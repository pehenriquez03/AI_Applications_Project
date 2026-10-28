"""Pure helpers for the Streamlit app (no streamlit import, so they are unit-testable)."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

STAGE_NAMES = [(1, "Read paper"), (2, "Extract strategy"), (3, "Validate and adapt"),
               (4, "Check data"), (5, "Construct portfolios"), (6, "Backtest"),
               (7, "Evaluate"), (8, "Monte Carlo"), (9, "Dossier")]


# ----------------------------------------------------------------------------
# file access
# ----------------------------------------------------------------------------
def read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # missing, partially written, or invalid
        return None


def read_progress(run_dir: Path) -> Optional[dict]:
    """Progress as a plain dict with exactly nine stage dicts, or None."""
    d = read_json(Path(run_dir) / "progress.json")
    if not isinstance(d, dict):
        return None
    have = {s.get("n"): s for s in d.get("stages", []) if isinstance(s, dict)}
    d["stages"] = [{"n": n, "name": name, "status": "pending", "detail": "", **have.get(n, {})}
                   for n, name in STAGE_NAMES]
    return d


def run_state(p: Optional[dict], job_active: bool = False) -> str:
    """finished | blocked | failed | running | incomplete | missing"""
    if not p:
        return "missing"
    st = [s["status"] for s in p["stages"]]
    rec = (p.get("recommendation") or "").lower()
    if job_active and "running" in st:
        return "running"
    if rec == "unable to evaluate" or "blocked" in st:
        return "blocked"
    if "failed" in st or p.get("error"):
        return "failed"
    if st[-1] == "done" or rec:
        return "finished"
    return "running" if job_active else "incomplete"


def short_title(title: str, n: int = 44) -> str:
    title = " ".join((title or "").split())
    return title if len(title) <= n else title[: n - 1].rstrip() + "…"


def paper_label(meta: Optional[dict], known: Optional[dict], fallback: str = "") -> str:
    """Short title from the registry when the paper's sha/file matches, else the title guess.

    known: {"sha": {sha256: label}, "file": {file_name: label}} (see papers_registry.known_papers).
    """
    meta = meta or {}
    known = known or {}
    label = (known.get("sha") or {}).get(meta.get("sha256")) or (known.get("file") or {}).get(meta.get("file_name"))
    if not label:
        label = meta.get("title_guess") or fallback
    label = re.sub(r"\s*\(decoy\)\s*$", "", (label or "").replace(" — mock paper", "")).strip(" :")
    m = re.match(r"^([^,&—]+?)\s*[,&].*?(\d{4}) — (.+)$", label)      # "A, B & C 2011 — topic" -> "A 2011 — topic"
    return short_title(f"{m.group(1)} {m.group(2)} — {m.group(3)}" if m else label, 52)


def list_runs(runs_dir: Path, known: Optional[dict] = None) -> list[dict]:
    """Run folders with a progress.json, newest first; excludes names starting '_' or 'mock'
    (demo_*, dev_* and everything else are listed)."""
    out = []
    if not Path(runs_dir).is_dir():
        return out
    for d in Path(runs_dir).iterdir():
        if not d.is_dir() or d.name.startswith("_") or d.name.startswith("mock"):
            continue
        f = d / "progress.json"
        if not f.exists():
            continue
        p = read_progress(d)
        if p is None:
            continue
        state = run_state(p)
        label = (p.get("recommendation") or {"failed": "failed", "blocked": "blocked",
                                             "incomplete": "incomplete"}.get(state, state))
        meta = read_json(d / "paper_meta.json")
        title = paper_label(meta if isinstance(meta, dict) else None, known, p.get("paper_title") or d.name)
        out.append(dict(id=d.name, title=title, label=label, mtime=f.stat().st_mtime))
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def api_key_status(root: Path, environ: dict) -> str:
    """'key found' if ANTHROPIC_API_KEY is in the environment or .env, else 'cached mode'. Never returns the key."""
    if environ.get("ANTHROPIC_API_KEY"):
        return "key found"
    try:
        for line in (Path(root) / ".env").read_text().splitlines():
            m = re.match(r"\s*(?:export\s+)?ANTHROPIC_API_KEY\s*=\s*(.+)", line)
            if m and m.group(1).strip().strip("'\""):
                return "key found"
    except Exception:
        pass
    return "cached mode"


# ----------------------------------------------------------------------------
# formatting and generic nested-dict tables
# ----------------------------------------------------------------------------
def fmt(v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "–"
        return f"{v:,.0f}" if abs(v) >= 1e4 else f"{v:.4g}"
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)) and all(not isinstance(x, (dict, list)) for x in v):
        return ", ".join(fmt(x) for x in v)
    return json.dumps(v, default=str)


def _scalar(v: Any) -> bool:
    return not isinstance(v, (dict, list))


def pretty(key: str) -> str:
    return str(key).replace("_", " ")


def table_sections(obj: Any, path: str = "") -> list[tuple[str, pd.DataFrame]]:
    """Render any nested structure of numbers as (title, all-string DataFrame) sections."""
    title = path
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj):
            df = pd.DataFrame([{pretty(k): fmt(v) for k, v in x.items()} for x in obj])
            return [(title, df)]
        return [(title, pd.DataFrame({"value": [fmt(x) for x in obj]}))] if obj else []
    if not isinstance(obj, dict):
        return [(title, pd.DataFrame({"value": [fmt(obj)]}))]
    out: list[tuple[str, pd.DataFrame]] = []
    scalars = {k: v for k, v in obj.items() if _scalar(v)}
    kids = {k: v for k, v in obj.items() if not _scalar(v)}
    if scalars:
        out.append((title, pd.DataFrame({"value": [fmt(v) for v in scalars.values()]},
                                        index=[pretty(k) for k in scalars])))
    if kids and all(isinstance(v, dict) and v and all(_scalar(x) for x in v.values()) for v in kids.values()):
        df = pd.DataFrame({pretty(k): {pretty(a): fmt(b) for a, b in v.items()} for k, v in kids.items()}).T
        out.append((title, df.fillna("–")))
    else:
        for k, v in kids.items():
            out.extend(table_sections(v, f"{path} / {pretty(k)}" if path else pretty(k)))
    return out


# ----------------------------------------------------------------------------
# extraction tab
# ----------------------------------------------------------------------------
EVIDENCE_PLAIN = {"sort_rule": "sort_rule_evidence", "weighting": "weighting_evidence",
                  "expected_sign": "expected_sign_evidence"}


def field_value_and_evidence(spec: dict, field: str) -> tuple[str, Optional[int], str]:
    v = spec.get(field)
    if isinstance(v, dict) and "value" in v:
        ev = v.get("evidence") or {}
        return fmt(v["value"]), ev.get("page"), ev.get("quote", "")
    ev = spec.get(EVIDENCE_PLAIN.get(field, ""), None) or {}
    return fmt(v), ev.get("page"), ev.get("quote", "")


def evidence_ok(validation: Optional[dict], field: str) -> Optional[bool]:
    """Best-effort read of validation.json['evidence'] for one field; None when unknown."""
    ev = (validation or {}).get("evidence")
    item = None
    if isinstance(ev, dict):
        item = ev.get(field)
        if item is None and isinstance(ev.get("fields"), dict):
            item = ev["fields"].get(field)
    elif isinstance(ev, list):
        for x in ev:
            if isinstance(x, dict) and x.get("field") == field:
                item = x
                break
    if isinstance(item, bool):
        return item
    if isinstance(item, dict):
        for k in ("valid", "ok", "found", "verified", "match", "matched"):
            if isinstance(item.get(k), bool):
                return item[k]
    return None


def agreement_text(agreement: Any, field: str) -> str:
    if not isinstance(agreement, dict) or field not in agreement:
        return ""
    a = agreement[field]
    if isinstance(a, bool):
        return "yes" if a else "no"
    if isinstance(a, (int, float)):
        return f"{a:.0%}" if 0 <= a <= 1 else fmt(a)
    return fmt(a)


# ----------------------------------------------------------------------------
# economics histograms
# ----------------------------------------------------------------------------
def histograms(mc: pd.DataFrame, needle: str, bins: int = 30, max_series: int = 4,
               cols: Optional[list] = None) -> Optional[pd.DataFrame]:
    """Shared-bin histogram table (column "bin" = bin centre) for the numeric columns named in `cols`,
    else all numeric columns whose name contains `needle`."""
    if cols is not None:
        cols = [c for c in cols if c in mc.columns and pd.api.types.is_numeric_dtype(mc[c])]
    else:
        cols = [c for c in mc.columns if needle in str(c).lower() and pd.api.types.is_numeric_dtype(mc[c])][:max_series]
    if not cols:
        return None
    data = mc[cols].dropna(how="all")
    unit = ""
    if float(data.abs().max().max()) >= 1e5:      # money columns: show in millions
        data, unit = data / 1e6, "millions"
    lo, hi = float(data.min().min()), float(data.max().max())
    if not math.isfinite(lo) or not math.isfinite(hi):
        return None
    if hi == lo:
        hi = lo + 1.0
    import numpy as np
    edges = np.linspace(lo, hi, bins + 1)
    centres = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    df = pd.DataFrame({str(c): np.histogram(data[c].dropna(), bins=edges)[0] for c in cols})
    df.insert(0, "bin", [round(c, 2) for c in centres])
    df.attrs["unit"] = unit
    return df


# ----------------------------------------------------------------------------
# evaluation / economics tables (all return all-string DataFrames; tolerate missing keys)
# ----------------------------------------------------------------------------
def pct(v: Any, d: int = 1) -> str:
    return "–" if not isinstance(v, (int, float)) or isinstance(v, bool) or math.isnan(v) else f"{v * 100:.{d}f}%"


def num(v: Any, d: int = 2) -> str:
    return "–" if not isinstance(v, (int, float)) or isinstance(v, bool) or math.isnan(v) else f"{v:,.{d}f}"


def money_m(v: Any, d: int = 2) -> str:
    """USD in millions."""
    return "–" if not isinstance(v, (int, float)) or isinstance(v, bool) or math.isnan(v) else f"{v / 1e6:,.{d}f}"


def _d(x: Any) -> dict:
    return x if isinstance(x, dict) else {}


def _frame(rows: dict[str, list[str]], cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(list(rows.values()), index=list(rows.keys()), columns=cols)


PERF_ROWS = [("strategy_gross", "Strategy, gross"), ("strategy_net", "Strategy, net of costs"),
             ("spy", "SPY"), ("ew_universe", "Equal-weight universe")]


def performance_table(s: dict) -> Optional[pd.DataFrame]:
    perf = _d(s.get("performance"))
    rows = {lab: [pct(_d(perf[k]).get("cagr")), pct(_d(perf[k]).get("vol")), num(_d(perf[k]).get("sharpe")),
                  pct(_d(perf[k]).get("max_drawdown")), num(_d(perf[k]).get("end_wealth"))]
            for k, lab in PERF_ROWS if k in perf}
    return _frame(rows, ["CAGR", "Volatility", "Sharpe", "Max drawdown", "Growth of 1"]) if rows else None


def relative_table(s: dict) -> Optional[pd.DataFrame]:
    rel = _d(s.get("relative"))
    names = {"vs_spy": "Net strategy minus SPY", "vs_ew_universe": "Net strategy minus equal-weight universe"}
    rows = {}
    for k, lab in names.items():
        r = _d(rel.get(k))
        if r:
            rows[lab] = [pct(r.get("annualized_diff")), pct(r.get("cagr_diff")), num(r.get("nw_t")),
                         f"{pct(r.get('ci95_low'), 2)} to {pct(r.get('ci95_high'), 2)}", fmt(r.get("n_months"))]
    return _frame(rows, ["Annualized diff", "CAGR diff", "Newey-West t", "95% CI, monthly", "Months"]) if rows else None


def alpha_table(s: dict) -> Optional[pd.DataFrame]:
    rows = {}
    for k, lab in (("alpha", "Four-factor alpha (vs the market and size, value, momentum)"),
                   ("spanning", "Spanning alpha (net strategy minus equal-weight universe)")):
        a = _d(s.get(k))
        if not a:
            continue
        b, t = _d(a.get("betas")), _d(a.get("beta_t"))
        rows[lab] = [pct(a.get("alpha_annual")), num(a.get("alpha_t"))] + \
                    [f"{num(b.get(f))} ({num(t.get(f), 1)})" if f in b else "–" for f in ("mkt_rf", "smb", "hml", "mom")] + \
                    [num(a.get("r2")), fmt(a.get("n_obs"))]
    return _frame(rows, ["Alpha, annual", "Alpha t", "Market (t)", "SMB (t)", "HML (t)", "MOM (t)", "R²", "Days"]) if rows else None


def cost_grid_table(s: dict) -> Optional[pd.DataFrame]:
    g = _d(_d(s.get("robustness")).get("cost_grid"))
    if not g:
        return None
    order = sorted(g, key=lambda k: float(k) if str(k).replace(".", "", 1).isdigit() else 1e9)
    rows = {f"{k} bps": [pct(_d(g[k]).get("cagr")), num(_d(g[k]).get("sharpe")), pct(_d(g[k]).get("alpha_annual")),
                         num(_d(g[k]).get("alpha_t"))] for k in order}
    return _frame(rows, ["CAGR", "Sharpe", "Alpha, annual", "Alpha t"])


def subperiod_table(s: dict) -> Optional[pd.DataFrame]:
    sp = _d(_d(s.get("robustness")).get("subperiods"))
    if not sp:
        return None
    rows = {f"{k}: {_d(v).get('start', '?')} to {_d(v).get('end', '?')}":
            [pct(_d(v).get("cagr")), num(_d(v).get("sharpe")), pct(_d(v).get("rel_ew_mean_monthly"), 2),
             num(_d(v).get("rel_ew_t"))] for k, v in sp.items() if isinstance(v, dict)}
    return _frame(rows, ["CAGR", "Sharpe", "Monthly diff vs equal-weight", "t"]) if rows else None


def groups_table(s: dict) -> tuple[Optional[pd.DataFrame], str]:
    """(table of ranked groups, one-line caption with rho and favored-minus-disfavored)."""
    g = _d(_d(s.get("robustness")).get("groups"))
    rows = [r for r in g.get("rows") or [] if isinstance(r, dict)]
    df = None
    if rows:
        df = pd.DataFrame({"Group": [fmt(r.get("group")) for r in rows],
                           "Mean monthly": [pct(r.get("mean_monthly"), 2) for r in rows],
                           "CAGR": [pct(r.get("cagr_daily")) for r in rows],
                           "Annualized mean": [pct(r.get("annualized_mean_x12")) for r in rows]})
    cap = []
    if "rho" in g:
        d = g.get("rho_in_expected_direction")
        cap.append(f"Spearman rho of group rank and return {num(g.get('rho'))} (p {num(g.get('rho_p'), 3)}); "
                   f"{'in' if d else 'not in' if d is False else 'direction not assessed against'} the expected direction.")
    fd = _d(g.get("favored_minus_disfavored"))
    if fd:
        cap.append(f"Group {fmt(fd.get('favored_group'))} minus group {fmt(fd.get('disfavored_group'))}: "
                   f"{pct(fd.get('mean_monthly_diff'), 2)} per month, Newey-West t {num(fd.get('nw_t'))}.")
    return df, " ".join(cap)


def implementation_table(s: dict) -> Optional[pd.DataFrame]:
    t, c, h, cap = (_d(s.get(k)) for k in ("turnover", "costs", "holdings", "capacity"))
    items = []
    if t:
        items += [("Mean monthly one-way turnover", pct(t.get("mean_monthly"), 0)), ("Peak monthly turnover", pct(t.get("max_monthly"), 0))]
    if c:
        items += [("Trading costs, share of average value", pct(c.get("pct_of_avg_value"))),
                  ("Trading costs, annualized", pct(c.get("annualized_pct_of_avg_value"), 2))]
    if h:
        items += [("Mean holdings", num(h.get("mean_n_holdings"), 0)), ("Largest single weight", pct(h.get("max_weight")))]
    if cap:
        pt = _d(cap.get("per_trade"))
        items += [("Fund size assumed", "$" + money_m(cap.get("fund_aum"), 0) + " million"),
                  ("Median trade, share of 20-day volume", pct(pt.get("median"))),
                  ("Largest trade, share of 20-day volume", pct(pt.get("max"))),
                  ("Trades above 10% of volume", pct(cap.get("frac_trades_over_10pct_adv")))]
    return pd.DataFrame({"Value": [v for _, v in items]}, index=[k for k, _ in items]) if items else None


def fractions_table(e: dict) -> Optional[pd.DataFrame]:
    items = [("Paths beating SPY", pct(e.get("frac_beats_spy"))), ("Paths beating the equal-weight universe", pct(e.get("frac_beats_ew"))),
             ("Paths with positive NPV", pct(e.get("frac_npv_positive"))),
             ("Paths with positive NPV, history demeaned", pct(e.get("frac_npv_positive_demeaned")))]
    items = [(k, v) for (k, v), key in zip(items, ("frac_beats_spy", "frac_beats_ew", "frac_npv_positive", "frac_npv_positive_demeaned"))
             if key in e]
    return pd.DataFrame({"Value": [v for _, v in items]}, index=[k for k, _ in items]) if items else None


def money_table(e: dict) -> Optional[pd.DataFrame]:
    se = e.get("expected_incremental_fees_pv_se")
    items = []
    if "expected_incremental_fees_pv" in e:
        v = money_m(e["expected_incremental_fees_pv"]) + (f" (se {money_m(se)})" if se is not None else "")
        items.append(("Expected incremental fees, present value", v))
    if "expected_npv" in e:
        items.append(("Expected NPV after research cost", money_m(e["expected_npv"])))
    if "research_cost_total" in e:
        items.append(("Research cost", money_m(e["research_cost_total"], 3)))
    if "breakeven_aum" in e:
        items.append(("Breakeven fund size", "none found" if e["breakeven_aum"] is None else "$" + money_m(e["breakeven_aum"], 0) + " million"))
    return pd.DataFrame({"USD millions": [v for _, v in items]}, index=[k for k, _ in items]) if items else None


QUANTILE_LABELS = {"end_wealth_strategy": ("Ending wealth, strategy (net of fees)", "x"), "end_wealth_ew": ("Ending wealth, equal-weight (net of fees)", "x"),
                   "end_wealth_spy": ("Ending wealth, SPY", "x"), "maxdd_strategy": ("Max drawdown, strategy", "%"),
                   "fees_strategy": ("Fees, strategy (USD m)", "m"), "fees_ew": ("Fees, equal-weight (USD m)", "m"),
                   "incremental_fees_pv": ("Incremental fees, PV (USD m)", "m"), "npv": ("NPV (USD m)", "m"),
                   "npv_demeaned": ("NPV, history demeaned (USD m)", "m"),
                   "incremental_fees_pv_demeaned": ("Incremental fees, PV, demeaned (USD m)", "m")}


def quantiles_table(e: dict) -> Optional[pd.DataFrame]:
    q = _d(e.get("quantiles"))
    if not q:
        return None
    cols = ["p05", "p25", "p50", "p75", "p95"]
    f = {"x": lambda v: num(v), "%": lambda v: pct(v), "m": lambda v: money_m(v)}
    rows = {}
    for k, v in q.items():
        lab, kind = QUANTILE_LABELS.get(k, (pretty(k), "x"))
        rows[lab] = [f[kind](_d(v).get(c)) for c in cols]
    return _frame(rows, ["5th", "25th", "median", "75th", "95th"])


def block_table(e: dict) -> Optional[pd.DataFrame]:
    b = _d(e.get("block_sensitivity"))
    if not b:
        return None
    return pd.DataFrame({"Paths with positive NPV": [pct(v) for v in b.values()]},
                        index=[f"{k}-day blocks" for k in b])
