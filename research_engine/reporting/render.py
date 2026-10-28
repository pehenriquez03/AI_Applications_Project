"""HTML dossier, blocked dossier and runs index (Jinja2 templates in ./templates).

Every page is a single self-contained file: inline CSS, charts as base64 PNGs
(matplotlib Agg), no external assets. Links to run files are relative.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

from research_engine import contracts as C  # noqa: E402
from research_engine.contracts import RunFiles  # noqa: E402

from .results import dig  # noqa: E402

log = logging.getLogger("engine.reporting")
TEMPLATES = Path(__file__).resolve().parent / "templates"

# chart colors: reference categorical slots 1-3 (validated all-pairs), muted greys for context
COL = {"strategy": "#2a78d6", "ew": "#eb6834", "spy": "#1baf7a", "grey": "#9a9893", "ink": "#52514e",
       "grid": "#e6e5e0"}
SERIES_LABEL = {"strategy_net": "Strategy (net)", "ew_universe": "EW universe", "spy": "SPY"}
SERIES_COL = {"strategy_net": COL["strategy"], "ew_universe": COL["ew"], "spy": COL["spy"]}


# ----------------------------------------------------------------------------
# formatting filters
# ----------------------------------------------------------------------------
def _num(x) -> Optional[float]:
    try:
        if x is None or isinstance(x, bool):
            return None
        x = float(x)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


def f_pct(x, d=1):
    x = _num(x)
    return "–" if x is None else f"{x * 100:.{d}f}%"


def f_num(x, d=2):
    x = _num(x)
    if x is None:
        return "–"
    if d == 0 or (abs(x) >= 1000 and float(x).is_integer()):
        return f"{x:,.0f}"
    return f"{x:,.{d}f}"


def f_money(x):
    x = _num(x)
    if x is None:
        return "–"
    a, s = abs(x), ("-" if x < 0 else "")
    if a >= 1e9:
        return f"{s}${a / 1e9:.2f}B"
    if a >= 1e6:
        return f"{s}${a / 1e6:.1f}M"
    if a >= 1e3:
        return f"{s}${a / 1e3:.0f}k"
    return f"{s}${a:.2f}"


def f_val(x):
    """Generic cell: lists joined, floats trimmed, None as dash."""
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f_num(x, 4 if abs(x) < 1 else 2)
    if isinstance(x, list):
        return ", ".join(f_val(v) for v in x)
    if isinstance(x, dict):
        return "; ".join(f"{k}: {f_val(v)}" for k, v in x.items())
    return str(x)


def f_size(n):
    n = float(n)
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


_PCT_KEYS = ("cagr", "return", "vol", "drawdown", "alpha_ann", "excess", "frac", "rate", "pct", "weight",
             "drag", "turnover", "agreement")
_MONEY_KEYS = ("fees", "npv", "aum", "cost_total", "usd", "breakeven")


def f_auto(x, key: str = ""):
    """Format by key name: fractions as %, money as $, t-stats/ratios as numbers, counts as integers."""
    k = str(key).lower()
    if _num(x) is None:
        return f_val(x)
    toks = set(k.replace("-", "_").split("_"))
    if k.endswith("_t") or k in ("t", "nw_t") or toks & {"stat", "beta", "sharpe", "rho", "r2", "ratio", "p"}:
        return f_num(x, 2) if "p" not in toks else f_num(x, 4)
    if toks & {"n", "names", "count", "bps", "days", "lag", "group", "paths", "seed", "obs", "calls", "tokens", "errors"}:
        return f_num(x, 1) if not float(x).is_integer() else f_num(x, 0)
    if any(t in k for t in _MONEY_KEYS):
        return f_money(x)
    if any(t in k for t in _PCT_KEYS):
        return f_pct(x, 2)
    return f_val(x)


def _env() -> Environment:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html", "j2"]),
                      trim_blocks=True, lstrip_blocks=True)
    env.filters.update(pct=f_pct, num=f_num, money=f_money, val=f_val, size=f_size, auto=f_auto)
    env.globals["dig"] = dig
    return env


def _css() -> str:
    return (TEMPLATES / "style.css").read_text()


# ----------------------------------------------------------------------------
# charts
# ----------------------------------------------------------------------------
def _style(ax, title: str):
    ax.set_title(title, loc="left", fontsize=10.5, color="#0b0b0b", pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(COL["grid"])
    ax.tick_params(colors=COL["ink"], labelsize=8.5, length=0)
    ax.grid(axis="y", color=COL["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _legend(ax):
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(COL["ink"])


def chart_growth(ret: pd.DataFrame) -> Optional[str]:
    cols = [c for c in ("strategy_net", "ew_universe", "spy") if c in ret]
    if not cols:
        return None
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    for c in cols:
        w = (1 + ret[c].fillna(0)).cumprod()
        ax.plot(w.index, w.values, color=SERIES_COL[c], lw=2 if c == "strategy_net" else 1.4, label=SERIES_LABEL[c])
        ax.annotate(f"{w.iloc[-1]:.2f}x", (w.index[-1], w.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=COL["ink"], va="center")
    _style(ax, "Growth of $1 (daily, net of costs for the strategy)")
    _legend(ax)
    return _png(fig)


def chart_drawdown(ret: pd.DataFrame) -> Optional[str]:
    cols = [c for c in ("strategy_net", "ew_universe") if c in ret]
    if not cols:
        return None
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    for c in cols:
        w = (1 + ret[c].fillna(0)).cumprod()
        dd = w / w.cummax() - 1
        ax.plot(dd.index, dd.values * 100, color=SERIES_COL[c], lw=1.6 if c == "strategy_net" else 1.2,
                label=SERIES_LABEL[c])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _style(ax, "Drawdown from previous peak")
    leg = ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    for t in leg.get_texts():
        t.set_color(COL["ink"])
    return _png(fig)


def chart_groups(table: list, favored: Optional[int], rho) -> Optional[str]:
    rows = [r for r in table or [] if _num(r.get("cagr")) is not None]
    if not rows:
        return None
    g = [int(r.get("group")) for r in rows]
    v = [float(r["cagr"]) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 2.8))
    colors = [COL["strategy"] if gi == favored else "#c9c7c0" for gi in g]
    ax.bar(g, v, color=colors, width=0.72, edgecolor="white", linewidth=2)
    ax.axhline(0, color=COL["ink"], lw=0.8)
    for gi, vi in zip(g, v):
        ax.annotate(f"{vi:.1f}", (gi, vi), xytext=(0, 3 if vi >= 0 else -10), textcoords="offset points",
                    ha="center", fontsize=7.5, color=COL["ink"])
    ax.margins(y=0.18)
    ax.set_xticks(g)
    ax.set_xticklabels([f"G{gi}" for gi in g])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    rt = f"; Spearman rho {float(rho):.2f}" if _num(rho) is not None else ""
    _style(ax, f"Net CAGR by ranked group (1 = lowest signal; favored group highlighted{rt})")
    return _png(fig)


def chart_hist(values: pd.Series, title: str, money: bool = True, ref: Optional[float] = 0.0,
               color: str = COL["strategy"]) -> Optional[str]:
    v = pd.Series(values).dropna()
    if v.empty:
        return None
    scale, unit = (1e6, "$M") if money and v.abs().max() >= 1e5 else ((1e3, "$k") if money else (1, ""))
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    ax.hist(v / scale, bins=40, color=color, edgecolor="white", linewidth=0.6)
    if ref is not None:
        ax.axvline(ref / scale, color="#0b0b0b", lw=1, ls="--")
    med = float(v.median()) / scale
    ax.axvline(med, color=COL["ink"], lw=1)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.annotate(f"median {med:,.2f}{(' ' + unit) if unit else ''}", (med, ax.get_ylim()[1] * 0.95),
                xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=COL["ink"], va="top")
    if unit:
        ax.set_xlabel(unit, fontsize=8, color=COL["ink"])
    _style(ax, title)
    return _png(fig)


def chart_wealth(mc: pd.DataFrame) -> Optional[str]:
    cols = [c for c in ("end_wealth_strategy", "end_wealth_ew", "end_wealth_spy") if c in mc]
    if not cols:
        return None
    lab = {"end_wealth_strategy": ("Strategy fund (net)", COL["strategy"]),
           "end_wealth_ew": ("EW universe fund (net)", COL["ew"]),
           "end_wealth_spy": ("SPY (gross)", COL["spy"])}
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    lo, hi = np.nanpercentile(mc[cols].values, [0.5, 99.5])
    bins = np.linspace(lo, hi, 40)
    for c in cols:
        ax.hist(mc[c].dropna(), bins=bins, histtype="step", lw=1.6, color=lab[c][1], label=lab[c][0])
    ax.axvline(1.0, color="#0b0b0b", lw=1, ls="--")
    ax.set_xlabel("ending wealth, multiple of start", fontsize=8, color=COL["ink"])
    _style(ax, "Investor ending wealth after 5 years")
    leg = ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(COL["ink"])
    return _png(fig)


def _safe_chart(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:  # noqa: BLE001  a chart failure never kills the dossier
        log.warning("chart %s failed: %s", fn.__name__, e)
        return None


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def code_version() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.ROOT, capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _downloads(run_dir: Path, extra: tuple[str, ...] = ()) -> list[dict]:
    names = {p.name for p in run_dir.iterdir() if p.is_file()} | set(extra)
    out = []
    for n in sorted(names):
        p = run_dir / n
        out.append({"name": n, "size": p.stat().st_size if p.exists() else None})
    return out


def _read(path: Path):
    try:
        return pd.read_parquet(path) if path.exists() else None
    except Exception as e:  # noqa: BLE001
        log.warning("could not read %s: %s", path, e)
        return None


def _write(path: Path, html: str) -> Path:
    path.write_text(html, encoding="utf-8")
    return path


def _footer(run_dir: Path, results: dict) -> dict:
    return {"run_id": results.get("run_id") or run_dir.name,
            "snapshot": dig(results, "data.snapshot") or C.SNAPSHOT_DIR.name,
            "code_version": code_version(),
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "model": _run_model(run_dir)}


def _run_model(run_dir: Path) -> str:
    """Model recorded in run_config.json (written by the pipeline); contracts.MODEL otherwise."""
    try:
        return json.loads((Path(run_dir) / "run_config.json").read_text()).get("model") or C.MODEL
    except Exception:  # noqa: BLE001
        return C.MODEL


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------
def render_dossier(run_dir, results: dict, narrative: Optional[dict]) -> Path:
    run_dir = Path(run_dir)
    ret = _read(run_dir / RunFiles.returns)
    mc = _read(run_dir / RunFiles.montecarlo)
    charts = {}
    if ret is not None:
        charts["growth"] = _safe_chart(chart_growth, ret)
        charts["drawdown"] = _safe_chart(chart_drawdown, ret)
    groups = dig(results, "robustness.groups") or {}
    table = groups.get("table")
    if not table and ret is not None:
        gcols = sorted([c for c in ret if c.startswith("group_")], key=lambda c: int(c.split("_")[1]))
        table = []
        for c in gcols:
            r = ret[c].dropna()
            table.append({"group": int(c.split("_")[1]),
                          "cagr": float((1 + r).prod() ** (252 / max(len(r), 1)) - 1) if len(r) else None})
    favored = groups.get("favored_group")
    if favored is None and table:
        fav = (dig(results, "adaptation.selection.favored") or "highest")
        gs = [int(r["group"]) for r in table]
        favored = min(gs) if fav == "lowest" else max(gs)
    charts["groups"] = _safe_chart(chart_groups, table, favored, groups.get("rho"))
    if mc is not None:
        charts["inc_fees"] = _safe_chart(chart_hist, mc.get("incremental_fees_pv"),
                                         "Incremental fee PV vs EW universe fund (headline)")
        charts["npv"] = _safe_chart(chart_hist, mc.get("npv"), "NPV after research cost", color=COL["ew"])
        charts["wealth"] = _safe_chart(chart_wealth, mc)
    narrative = narrative or {}
    html = _env().get_template("dossier.html.j2").render(
        r=results, n=narrative, charts=charts, group_table=table, favored_group=favored, css=_css(),
        downloads=_downloads(run_dir, (RunFiles.dossier,)), footer=_footer(run_dir, results), blocked=False,
        section_names=list((narrative.get("sections") or {}).keys()))
    return _write(run_dir / RunFiles.dossier, html)


def render_blocked(run_dir, reason: str) -> str:
    """Short dossier for a blocked run; writes results.json (if possible) and dossier.html; returns summary."""
    from .results import assemble_results
    run_dir = Path(run_dir)
    try:
        results = assemble_results(run_dir)
    except Exception as e:  # noqa: BLE001
        log.warning("assemble_results failed on blocked run: %s", e)
        results = {"run_id": run_dir.name}
    results["status"] = "blocked"
    results["status_reasons"] = [reason]
    results["recommendation"] = "unable to evaluate"
    results["recommendation_reasons"] = [f"Unable to evaluate: {reason}"]
    title = dig(results, "paper.title") or run_dir.name
    summary = f"Unable to evaluate. {title}: {reason}"
    try:
        (run_dir / RunFiles.results).write_text(json.dumps(results, indent=1, default=str))
    except Exception as e:  # noqa: BLE001
        log.warning("could not write results.json: %s", e)
    html = _env().get_template("dossier.html.j2").render(
        r=results, n={}, charts={}, group_table=None, favored_group=None, css=_css(),
        downloads=_downloads(run_dir, (RunFiles.dossier,)), footer=_footer(run_dir, results), blocked=True,
        reason=reason, section_names=[])
    _write(run_dir / RunFiles.dossier, html)
    return summary


def _index_row(d: Path) -> dict:
    prog = {}
    try:
        prog = json.loads((d / RunFiles.progress).read_text())
    except Exception:  # noqa: BLE001
        pass
    res = {}
    try:
        res = json.loads((d / RunFiles.results).read_text())
    except Exception:  # noqa: BLE001
        pass
    meta = {}
    try:
        meta = json.loads((d / RunFiles.paper_meta).read_text())
    except Exception:  # noqa: BLE001
        pass
    interp = res.get("interpretation") or {}
    agreement = interp.get("agreement_rate")
    if agreement is None:
        agreement = interp.get("mean_field_agreement")
    stages = prog.get("stages") or []
    return {"run_id": d.name,
            "title": prog.get("paper_title") or dig(res, "paper.title") or meta.get("title") or d.name,
            "slug": prog.get("strategy_slug") or res.get("strategy_slug") or "",
            "recommendation": prog.get("recommendation") or res.get("recommendation") or
                              ("failed" if prog.get("error") else "in progress"),
            "net_cagr": dig(res, "performance.headline.strategy_net_cagr"),
            "alpha_t": dig(res, "performance.headline.alpha_t"),
            "p_npv": dig(res, "economics.frac_npv_positive"),
            "agreement": agreement, "preliminary": bool(interp.get("preliminary")),
            "has_dossier": (d / RunFiles.dossier).exists(),
            "stages_done": sum(s.get("status") == "done" for s in stages), "n_stages": len(stages)}


def render_index(runs_dir) -> Path:
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    rows = [_index_row(d) for d in sorted(runs_dir.iterdir())
            if d.is_dir() and (d / RunFiles.progress).exists()]
    rows.sort(key=lambda r: (str(r["title"]).lower(), r["run_id"]))
    html = _env().get_template("index.html.j2").render(
        rows=rows, css=_css(), generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        code_version=code_version())
    corr = _correlation_block(runs_dir, rows)
    if corr and "</body>" in html:
        html = html.replace("</body>", corr + "\n</body>", 1)
    return _write(runs_dir / "index.html", html)


def _correlation_block(runs_dir: Path, rows: list) -> str:
    """Monthly net-return correlation across completed strategies (one run per strategy slug,
    newest wins), rendered as a small table. Five of the six papers consume past prices, so this
    is how much independent evidence the set holds."""
    import pandas as pd
    series = {}
    for r in rows:
        d = runs_dir / r["run_id"]
        f = d / RunFiles.returns
        if not f.exists() or not r.get("slug"):
            continue
        try:
            rr = pd.read_parquet(f)
            x = rr["strategy_net"] - rr["ew_universe"]   # excess over the EW universe: strips the shared market move
            series[r["slug"]] = x.resample("ME").sum()
        except Exception:  # noqa: BLE001
            continue
    if len(series) < 2:
        return ""
    df = pd.DataFrame(series).dropna()
    c = df.corr().round(2)
    head = "".join(f"<th>{k}</th>" for k in c.columns)
    body = "".join("<tr><th>" + i + "</th>" + "".join(f"<td>{c.at[i, j]:.2f}</td>" for j in c.columns) + "</tr>"
                   for i in c.index)
    return ('<section class="card"><h2>Monthly excess-return correlation across strategies (net, minus the EW universe)</h2>'
            '<p class="muted">Equal-weighted, long-only versions on the same universe and sample, measured in excess of the equal-weighted universe so the shared market move is removed. High correlations '
            'mean the papers are not independent evidence.</p>'
            f'<table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table></section>')

