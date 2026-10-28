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


def list_runs(runs_dir: Path) -> list[dict]:
    """Run folders with a progress.json, newest first; excludes names starting '_' or 'mock'."""
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
        out.append(dict(id=d.name, title=short_title(p.get("paper_title") or d.name),
                        label=label, mtime=f.stat().st_mtime))
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
def histograms(mc: pd.DataFrame, needle: str, bins: int = 30, max_series: int = 4) -> Optional[pd.DataFrame]:
    """Shared-bin histogram table (column "bin" = bin centre) for all numeric columns whose name contains `needle`."""
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
