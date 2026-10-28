"""WARREN — your research screening agent. Terminal-style Streamlit front end.

Launch:  .venv/bin/streamlit run research_engine/app/streamlit_app.py
Layout:  top bar (name, tagline, last run, parameter chips) · left: the run (steps, performance,
         tabs, summary) · right: paper tiles (hover = preview), PDF drop, parameters, run, previous runs.
Reads run folders under runs/ (progress.json is polled while a run is active); runs the pipeline
in a background thread. The API key is only ever placed in the process environment.
"""
from __future__ import annotations

import hashlib
import html
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from research_engine.app import helpers as H
from research_engine.app.papers_registry import REGISTRY, abs_path, known_papers
from research_engine.contracts import (
    COST_BPS_BASE, COST_BPS_GRID, EVAL_END, EVAL_START, FUND, MATERIAL_FIELDS, MIN_HISTORY_DAYS,
    MIN_NAMES, MIN_NAMES_DOUBLE_SORT, MODEL, MONTE_CARLO, RESEARCH_COST, RUNS_DIR, SNAPSHOT_DIR,
    WARMUP_START,
)

APP_NAME = "WARREN"
TAGLINE = "YOUR RESEARCH SCREENING AGENT"
# Logo: a W drawn as a price line with a last-print dot (option 1 on the WARREN logo canvas).
LOGO_SVG = ('<svg class="logo" viewBox="0 0 40 40" role="img" aria-label="WARREN">'
            '<polyline points="4,9 12,31 20,17 28,31 36,9" fill="none" stroke="#f58220" stroke-width="5" '
            'stroke-linecap="round" stroke-linejoin="round"></polyline><circle cx="36" cy="9" r="3.6" fill="#f2f2f2"></circle></svg>')
ORANGE, GREEN, RED, PURPLE, GREY = "#f58220", "#3cc46a", "#e05252", "#c04bd6", "#8c8c8c"
MODELS = ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"]
MONTHS = [f"{y}-{m:02d}" for y in range(2013, 2026) for m in range(1, 13)]
DEFAULT_WINDOW = (EVAL_START[:7], EVAL_END[:7])
MIN_WINDOW_MONTHS = 24
FOLLOW = "__job__"          # selected_run sentinel: follow whichever run the job is on
NA = "not available yet"
SHORT = {"oil": "Aidan mock", "decoy_profitability": "Novy-Marx", "decoy_tsmom": "Moskowitz"}
CODES = {"max": "MAX", "momentum": "MOM", "seasonality": "SEAS", "lowvol": "IVOL", "high52": "52WH",
         "frog": "FROG", "oil": "OIL", "decoy_profitability": "GPROF", "decoy_tsmom": "TSMOM"}
LABEL_COLOR = {"pursue": "green", "investigate further": "orange", "deprioritize": "gray",
               "unable to evaluate": "red", "blocked": "red", "failed": "red"}

st.set_page_config(page_title=f"{APP_NAME} · {TAGLINE}", layout="wide", initial_sidebar_state="collapsed")

st.markdown(f"""<style>
.stApp, .stApp *:not([data-testid="stIconMaterial"]) {{ font-family: Arial, Helvetica, sans-serif; }}
.stApp code, .stApp pre, .stApp pre * {{ font-family: Menlo, Consolas, monospace; }}
.stApp {{ background: #111; color: #d9d9d9; font-size: 12.5px; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.stAppDeployButton {{ display: none; }}
.block-container {{ padding: .5rem 1.1rem 2rem; max-width: 100%; }}
.stApp p, .stApp li {{ font-size: 12.5px; }}
.stApp h3 {{ font-size: 15px; font-weight: bold; margin: .2rem 0 .1rem; }}
.stApp h4 {{ font-size: 12px; font-weight: bold; text-transform: uppercase; letter-spacing: .06em; color: #cfcfcf; }}
/* top bar */
.topbar {{ display: flex; align-items: flex-end; gap: 14px; border-bottom: 1px solid #2c2c2c; padding: 2px 0 6px; }}
.topbar .brand {{ display: flex; align-items: center; gap: 10px; }}
.topbar .brand .logo {{ width: 34px; height: 34px; flex: none; }}
.topbar .words {{ display: flex; flex-direction: column; line-height: 1.05; }}
.topbar .name {{ font-size: 26px; font-weight: bold; color: {ORANGE}; letter-spacing: .06em; }}
.topbar .tag {{ font-size: 10.5px; color: #bdbdbd; letter-spacing: .14em; text-transform: uppercase; margin-top: 3px; }}
.topbar .last {{ margin-left: auto; font-size: 12px; color: #bdbdbd; white-space: nowrap; }}
.topbar .last b {{ color: #f2f2f2; }}
.up {{ color: {GREEN}; }} .dn {{ color: {RED}; }} .amber {{ color: {ORANGE}; }} .mute {{ color: {GREY}; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 0; font-size: 11.5px; color: #bdbdbd; border-bottom: 1px solid #2c2c2c; padding: 5px 0; margin-bottom: 8px; }}
.chips .c {{ padding: 2px 10px; border-right: 1px solid #333; }}
.chips .c:first-child {{ padding-left: 0; }} .chips .c:last-child {{ border-right: 0; }}
.chips .k {{ color: {GREY}; margin-right: 5px; text-transform: uppercase; font-size: 10.5px; letter-spacing: .05em; }}
.chips .hl {{ background: {ORANGE}; color: #111; font-weight: bold; padding: 1px 6px; border-radius: 2px; }}
/* panels */
.ph {{ background: #232323; color: #e6e6e6; font-weight: bold; font-size: 11px; letter-spacing: .07em;
      text-transform: uppercase; padding: 4px 8px; margin: 8px 0 6px; border-left: 3px solid {ORANGE}; }}
.ph .r {{ float: right; color: {GREY}; font-weight: normal; letter-spacing: 0; text-transform: none; }}
.ph .chev {{ color: {ORANGE}; margin-left: 4px; }}
.kv {{ display: grid; grid-template-columns: 1fr auto; gap: 1px 10px; font-size: 11.5px; }}
.kv .k {{ color: {GREY}; }} .kv .v {{ color: #e0e0e0; text-align: right; font-variant-numeric: tabular-nums; }}
.kv .hl {{ color: {ORANGE}; font-weight: bold; }}
/* run header */
.runh {{ display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; margin: 4px 0 2px; }}
.runh .t {{ font-size: 16px; font-weight: bold; color: #f2f2f2; }}
.runh .win {{ background: {ORANGE}; color: #111; font-weight: bold; font-size: 11px; padding: 1px 7px; border-radius: 2px; }}
.quote {{ display: flex; gap: 18px; font-size: 12px; color: #bdbdbd; margin: 2px 0 6px; flex-wrap: wrap; }}
.quote .k {{ color: {GREY}; font-size: 10.5px; text-transform: uppercase; letter-spacing: .05em; margin-right: 4px; }}
.quote .v {{ color: #f2f2f2; font-weight: bold; font-variant-numeric: tabular-nums; }}
/* steps */
.steps {{ margin: .2rem 0 .6rem; font-variant-numeric: tabular-nums; font-size: 12px; }}
.step {{ display: flex; gap: .8rem; line-height: 1.6; border-bottom: 1px dotted #262626; }}
.step .n {{ width: 1.1rem; color: {GREY}; }}
.step .name {{ width: 11rem; flex: none; }}
.step .detail {{ flex: 1; color: #bdbdbd; }}
.s-done .name {{ color: #e0e0e0; }}
.s-running .name, .s-running .detail {{ color: {ORANGE}; }}
.s-pending {{ color: #5a5a5a; }} .s-pending .n {{ color: #444; }}
.s-blocked .name, .s-blocked .detail, .s-failed .name, .s-failed .detail {{ color: {RED}; }}
.err {{ color: {RED}; font-size: 12px; }}
/* widgets */
.stApp hr {{ border-color: #2c2c2c; margin: .8rem 0 .4rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid #2c2c2c; background: #161616; }}
.stTabs [data-baseweb="tab"] {{ padding: 4px 14px; height: 30px; }}
.stTabs [data-baseweb="tab"] p {{ font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {ORANGE}; }}
.stTabs [aria-selected="true"] p {{ color: {ORANGE}; }}
.stApp button[kind="primary"] {{ background: {ORANGE}; border-color: {ORANGE}; color: #111; font-weight: bold; border-radius: 2px; }}
.stApp button[kind="primary"]:hover {{ background: #ff9a3c; border-color: #ff9a3c; color: #111; }}
.stApp button[kind="secondary"], .stApp button[kind="tertiary"] {{ border-radius: 2px; }}
[data-testid="stFileUploaderDropzone"] {{ background: #161616; border: 1px dashed {ORANGE}; border-radius: 2px; }}
[data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span {{ color: #bdbdbd; }}
.stApp [data-testid="stExpander"] details {{ border: 1px solid #2c2c2c; border-radius: 2px; background: #161616; }}
.stApp [data-testid="stCaptionContainer"] p {{ color: {GREY}; font-size: 11.5px; }}
/* paper tiles: a 3-column grid of buttons; selected = primary */
.st-key-tiles [data-testid="stHorizontalBlock"] {{ gap: 5px; }}
.st-key-tiles [data-testid="column"] {{ padding: 0; min-width: 0; }}
.st-key-tiles .stButton {{ margin-bottom: 5px; }}
.st-key-tiles button {{ width: 100%; min-height: 50px; height: auto; padding: 5px 4px; border: 1px solid #353535;
                        background: #181818; color: #d9d9d9; border-radius: 2px; }}
.st-key-tiles button:hover {{ border-color: {ORANGE}; color: #fff; background: #1f1f1f; }}
.st-key-tiles button[kind="primary"] {{ background: #2b1c0d; color: {ORANGE}; border-color: {ORANGE}; }}
.st-key-tiles button p {{ font-size: 10.5px; line-height: 1.25; white-space: normal; overflow: hidden; text-overflow: ellipsis; }}
.st-key-tiles button p strong {{ font-size: 12.5px; letter-spacing: .04em; }}
.st-key-tiles button[kind="secondary"]:has(p em) {{ border-style: dashed; color: #9a9a9a; }}
/* previous runs: one line each */
.st-key-lines .stButton {{ margin: 0; }}
.st-key-lines [data-testid="stVerticalBlock"] {{ gap: 0; }}
.st-key-lines [data-testid="stElementContainer"] {{ margin: 0; }}
.st-key-lines button {{ width: 100%; justify-content: flex-start; text-align: left; padding: 1px 4px; min-height: 0;
                        border: 0; border-bottom: 1px solid #222; border-radius: 0; background: transparent; }}
.st-key-lines button:hover {{ background: #1c1c1c; }}
.st-key-lines button p {{ font-size: 11.5px; text-align: left; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
                          font-variant-numeric: tabular-nums; }}
.st-key-lines button p code {{ font-family: Menlo, Consolas, monospace; font-size: 10.5px; color: {GREY}; background: transparent; padding: 0; }}
.legend {{ font-size: 11px; color: {GREY}; margin: -2px 0 4px; }}
.runstate {{ font-size: 11.5px; color: #d9d9d9; padding-top: 7px; }}
.foot {{ border-top: 1px solid #2c2c2c; margin-top: 18px; padding-top: 8px; font-size: 11.5px; color: {GREY}; line-height: 1.5; }}
.foot b {{ color: #bdbdbd; }}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# background job (one at a time; survives browser refresh via cache_resource)
# ----------------------------------------------------------------------------
@st.cache_resource
def holder() -> dict:
    return {"job": None}


class Job:
    """tasks: dicts with label, pdf, experiment, prefix (sha12), run_dir, start_stage, model, eval_start, eval_end."""

    def __init__(self, tasks: list[dict]):
        self.tasks, self.i, self.started = tasks, 0, time.time()
        self.errors: list[str] = []
        self.active = True
        self.last_dir: Path | None = None

    def current_dir(self) -> Path | None:
        t = self.tasks[min(self.i, len(self.tasks) - 1)]
        if t.get("run_dir"):
            return Path(t["run_dir"])
        if not RUNS_DIR.is_dir():
            return None
        cands = [d for d in RUNS_DIR.iterdir()
                 if d.is_dir() and d.name.startswith(t["prefix"]) and d.stat().st_mtime >= self.started - 2]
        return max(cands, key=lambda d: d.name) if cands else None


def _work(job: Job) -> None:
    try:
        from research_engine.pipeline import run_pipeline
    except BaseException as e:  # noqa: BLE001
        job.errors.append(f"cannot import pipeline: {type(e).__name__}: {e}")
        job.active = False
        return
    for i, t in enumerate(job.tasks):
        job.i, job.started = i, time.time()
        bak = None
        opts = dict(llm="api", model=t.get("model"), eval_start=t.get("eval_start"), eval_end=t.get("eval_end"))
        try:
            if t.get("run_dir") and t.get("start_stage") == 9:       # regenerate narrative: keep the old one as a fallback
                nf = Path(t["run_dir"]) / "narrative.json"
                if nf.exists():
                    bak = nf.read_bytes()
                    nf.unlink()
                d = run_pipeline(Path(t["pdf"]), Path(t["run_dir"]), t.get("experiment") or None, start_stage=9, **opts)
                if bak is not None and (H.read_json(nf) or {}).get("source", "template") == "template":
                    nf.write_bytes(bak)                              # model unavailable: do not replace it with the template
                    job.errors.append(f"{t['label']}: narrative not regenerated (model unavailable); kept the previous one")
            else:
                d = run_pipeline(Path(t["pdf"]), run_dir=t.get("run_dir"),
                                 selected_experiment=t.get("experiment") or None,
                                 start_stage=t.get("start_stage", 1), **opts)
            job.last_dir = Path(d)
        except BaseException as e:  # noqa: BLE001 - recorded, next paper still runs
            job.errors.append(f"{t['label']}: {type(e).__name__}: {e}")
            job.last_dir = job.current_dir() or job.last_dir
            nf = Path(t["run_dir"]) / "narrative.json" if t.get("run_dir") else None
            if bak is not None and nf is not None and not nf.exists():
                nf.write_bytes(bak)
    job.active = False


def start_job(tasks: list[dict]) -> None:
    job = Job(tasks)
    holder()["job"] = job
    threading.Thread(target=_work, args=(job,), daemon=True).start()


# ----------------------------------------------------------------------------
# state and small helpers
# ----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("selected_run", None)
ss.setdefault("preview", None)
ss.setdefault("compare", False)
ss.setdefault("queue", [])          # [slug, ...] from the registry
ss.setdefault("upload_n", 0)

job: Job | None = holder()["job"]
active = bool(job and job.active)


def go(run=None, preview=None, compare=False) -> None:
    ss.selected_run, ss.preview, ss.compare = run, preview, compare


def esc(s: str) -> str:
    return html.escape(str(s)).replace("$", "&#36;")


def md_safe(s: str) -> str:
    return str(s).replace("$", "\\$")


def toggle_paper(slug: str) -> None:
    """One paper at a time: clicking a tile selects it (and replaces any other), clicking again clears it."""
    ss.queue = [] if slug in ss.queue else [slug]
    go(preview=slug)


def window_dates(win: tuple[str, str]) -> tuple[str, str]:
    """('2015-01', '2025-12') -> ('2015-01-01', '2025-12-31')."""
    a, b = win
    return f"{a}-01", str(pd.Period(b, freq="M").end_time.date())


def window_months(win: tuple[str, str]) -> int:
    return (pd.Period(win[1], freq="M") - pd.Period(win[0], freq="M")).n + 1


@st.cache_data(show_spinner=False)
def pdf_preview(path: str, mtime: float) -> tuple[int, str, str]:
    """(pages, pdf metadata title, first-page text excerpt) for the hover preview."""
    from pypdf import PdfReader
    r = PdfReader(path)
    text = (r.pages[0].extract_text() or "") if len(r.pages) else ""
    try:
        title = (r.metadata or {}).get("/Title") or ""
    except Exception:  # noqa: BLE001
        title = ""
    title = " ".join(str(title).split())
    if any(t in title.lower() for t in ("microsoft word", ".doc", ".pdf", ".tex", "untitled")) or len(title) < 8:
        title = ""                                   # a file name is not a title
    return len(r.pages), title, " ".join(text.split())[:700]


def paper_parts(label: str) -> tuple[str, str, str]:
    """'Bali, Cakici & Whitelaw 2011 — MAX' -> ('Bali, Cakici & Whitelaw', '2011', 'MAX')."""
    head, _, topic = label.partition(" — ")
    authors, _, year = head.rpartition(" ")
    return authors, year, topic.replace(" (decoy)", "")


def tile_help(label: str, rel: str) -> str:
    authors, year, topic = paper_parts(label)
    p = abs_path(rel)
    if not p.exists():
        return f"**{label}**\n\nFile not found: `{rel}`"
    try:
        n, title, text = pdf_preview(str(p), p.stat().st_mtime)
    except Exception as e:  # noqa: BLE001
        return f"**{label}**\n\nPreview unavailable: {type(e).__name__}"
    head = f"**{md_safe(title or topic)}**  \n{md_safe(authors)} · {year} · {n} pages"
    if "(decoy)" in label:
        head += "  \n*decoy: a correct extraction reports it as not implementable here*"
    return f"{head}\n\n{md_safe(text) or '(no extractable text on page 1)'}…"


def run_window(run_dir: Path) -> tuple[str, str, str, bool]:
    """(eval_start, eval_end, model, custom) for a run folder; defaults when the run predates run_config.json."""
    cfg = H.read_json(run_dir / "run_config.json") or {}
    exp = H.read_json(run_dir / "experiment_spec.json") or {}
    start = str(cfg.get("eval_start") or exp.get("eval_start") or EVAL_START)[:10]
    end = str(cfg.get("eval_end") or exp.get("eval_end") or EVAL_END)[:10]
    return start, end, cfg.get("model") or MODEL, bool(cfg.get("custom_window"))


# ----------------------------------------------------------------------------
# top bar
# ----------------------------------------------------------------------------
_manifest = H.read_json(SNAPSHOT_DIR / "manifest.json") or {}
_runs = H.list_runs(RUNS_DIR, known_papers())
_last = _runs[0] if _runs else None
_sel_window = tuple(ss.get("window", DEFAULT_WINDOW))
_sel_model = ss.get("model", MODEL)
_ws, _we = window_dates(_sel_window)
_key = H.api_key_status(ROOT, os.environ)

last_html = ""
if _last:
    col = {"green": "up", "orange": "amber", "red": "dn"}.get(LABEL_COLOR.get(_last["label"], "gray"), "mute")
    last_html = (f'<div class="last">Last <b>{esc(_last["title"])}</b> '
                 f'<span class="{col}">▲ {esc(_last["label"])}</span></div>')
st.markdown(f"""<div class="topbar"><div class="brand">{LOGO_SVG}<div class="words"><div class="name">{APP_NAME}</div>
<div class="tag">{TAGLINE}</div></div></div>{last_html}</div>
<div class="chips">
<div class="c"><span class="k">Model</span>{esc(_sel_model)}</div>
<div class="c"><span class="k">Universe</span>{_manifest.get("n_universe", "?")} S&amp;P 500 names, equal weight</div>
<div class="c"><span class="k">Backtest</span><span class="hl">{_ws} → {_we}</span></div>
<div class="c"><span class="k">Warm-up</span>from {WARMUP_START}</div>
<div class="c"><span class="k">Snapshot</span>{esc(_manifest.get("snapshot", "?"))}</div>
<div class="c"><span class="k">Cost</span>{COST_BPS_BASE} bps</div>
<div class="c"><span class="k">API</span>{esc(_key)}</div>
</div>""", unsafe_allow_html=True)

left, right = st.columns([2.55, 1], gap="medium")


# ----------------------------------------------------------------------------
# right column: papers, drop, parameters, run, previous runs
# ----------------------------------------------------------------------------
def panel(title: str, right_text: str = "") -> None:
    r = f'<span class="r">{esc(right_text)}</span>' if right_text else ""
    st.markdown(f'<div class="ph">{esc(title)}<span class="chev">›</span>{r}</div>', unsafe_allow_html=True)


with right:
    _n_months = window_months(tuple(ss.get("window", DEFAULT_WINDOW)))
    _uploads_pending = bool(ss.get(f"uploader_{ss.upload_n}"))
    _sel_label = next((l for l, _, sl in REGISTRY if sl == ss.queue[0]), "") if ss.queue else ""
    can_run = (ss.queue or _uploads_pending) and _n_months >= MIN_WINDOW_MONTHS and not active
    rc1, rc2 = st.columns([1, 1.6])
    run_clicked = rc1.button("Run", type="primary", disabled=not can_run, key="run_btn", width="stretch")
    if active:
        rc2.markdown('<div class="runstate amber">running… the button unlocks when the run finishes</div>', unsafe_allow_html=True)
    elif _sel_label:
        rc2.markdown(f'<div class="runstate">selected: <b>{esc(_sel_label.split(" — ")[0])}</b></div>', unsafe_allow_html=True)
    elif _uploads_pending:
        rc2.markdown('<div class="runstate">dropped PDF ready</div>', unsafe_allow_html=True)
    else:
        rc2.markdown('<div class="runstate mute">click one paper below, then Run</div>', unsafe_allow_html=True)

    panel("Papers", "selected: " + CODES.get(ss.queue[0], ss.queue[0]) if ss.queue else "click one, then press Run")
    with st.container(key="tiles"):
        for i in range(0, len(REGISTRY), 3):
            cols = st.columns(3)
            for c, (label, rel, slug) in zip(cols, REGISTRY[i:i + 3]):
                authors, year, topic = paper_parts(label)
                code = CODES.get(slug, slug.upper()[:5])
                first = SHORT.get(slug) or authors.split(",")[0].split(" &")[0]
                text = f"**{code}**  \n{first} {year}"
                if "(decoy)" in label:
                    text = f"**{code}**  \n*{first} {year}*"
                c.button(text, key=f"paper_{slug}", help=tile_help(label, rel),
                         type="primary" if slug in ss.queue else "secondary", on_click=toggle_paper, args=(slug,))
    st.markdown('<div class="legend">Click one paper and press Run; no upload needed. One paper at a time. '
                'Dashed = decoy (should come back not implementable).</div>', unsafe_allow_html=True)

    panel("Drop a PDF")
    uploads = st.file_uploader("PDF files to screen", type=["pdf"], accept_multiple_files=True,
                               key=f"uploader_{ss.upload_n}", label_visibility="collapsed")

    panel("Parameters")
    st.selectbox("Model", MODELS, index=MODELS.index(_sel_model) if _sel_model in MODELS else 0, key="model",
                 help="Extraction, check pass and narrative are written by this model. Cached answers exist "
                      "for the nine papers under claude-sonnet-5-5; any other choice calls the API.")
    st.select_slider("Backtest window", options=MONTHS, value=_sel_window, key="window",
                     help=f"Evaluation window. Signals warm up on all snapshot data from {WARMUP_START}; "
                          f"the window must hold at least {MIN_WINDOW_MONTHS} months.")
    n_months = window_months(tuple(ss.window))
    if n_months < MIN_WINDOW_MONTHS:
        st.markdown(f'<div class="err">Window too short: {n_months} months (minimum {MIN_WINDOW_MONTHS}).</div>',
                    unsafe_allow_html=True)
    experiment = st.text_input("Experiment (optional)", key="experiment",
                               placeholder="blank = the paper's main experiment")
    fund = FUND
    st.markdown(f"""<div class="kv">
<span class="k">Backtest window</span><span class="v hl">{_ws} → {_we} · {n_months} months</span>
<span class="k">Warm-up data from</span><span class="v">{WARMUP_START}</span>
<span class="k">Universe</span><span class="v">{_manifest.get("n_universe", "?")} names · survivor S&amp;P 500</span>
<span class="k">Weighting / rebalance</span><span class="v">equal · monthly</span>
<span class="k">Trading cost</span><span class="v">{COST_BPS_BASE} bps (grid {"/".join(map(str, COST_BPS_GRID))})</span>
<span class="k">Min names per leg</span><span class="v">{MIN_NAMES} (double sort {MIN_NAMES_DOUBLE_SORT})</span>
<span class="k">Min history</span><span class="v">{MIN_HISTORY_DAYS} trading days</span>
<span class="k">Benchmarks</span><span class="v">SPY · equal-weight universe</span>
<span class="k">Fund</span><span class="v">${fund["starting_aum"] / 1e6:.0f}M · {fund["management_fee_rate"]:.0%}/{fund["performance_fee_rate"]:.0%} · hurdle {fund["hurdle_rate"]:.0%} {fund["hurdle_type"]} · HWM</span>
<span class="k">Horizon / discount</span><span class="v">{fund["horizon_years"]} y · {fund["discount_rate"]:.0%}</span>
<span class="k">Monte Carlo</span><span class="v">{MONTE_CARLO["n_paths"]} paths · {MONTE_CARLO["mean_block_days"]}-day blocks</span>
<span class="k">Research cost</span><span class="v">{RESEARCH_COST["dev_review_hours"]} h + {RESEARCH_COST["maintenance_hours_per_year"]} h/y @ ${RESEARCH_COST["hourly_rate"]:.0f} + tokens</span>
</div>""", unsafe_allow_html=True)

    if run_clicked:
        tasks, problems = [], []
        es, ee = window_dates(tuple(ss.window))
        custom = tuple(ss.window) != DEFAULT_WINDOW
        common = dict(experiment=experiment, model=ss.model,
                      eval_start=es if custom else None, eval_end=ee if custom else None)
        for slug in ss.queue:
            label, rel, _ = next(r for r in REGISTRY if r[2] == slug)
            p = abs_path(rel)
            if not p.exists():
                problems.append(f"missing file: {rel}")
                continue
            tasks.append(dict(label=label, pdf=str(p), prefix=hashlib.sha256(p.read_bytes()).hexdigest()[:12], **common))
        for f in uploads or []:
            data = f.getvalue()
            sha = hashlib.sha256(data).hexdigest()
            dest = RUNS_DIR / "_uploads" / f"{sha}.pdf"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_bytes(data)
            tasks.append(dict(label=f.name, pdf=str(dest), prefix=sha[:12], **common))
        for pr in problems:
            st.markdown(f'<div class="err">{esc(pr)}</div>', unsafe_allow_html=True)
        if tasks:
            start_job(tasks)
            ss.queue = []
            ss.upload_n += 1
            go(run=FOLLOW)
            st.rerun()
    panel("Previous runs", f"{len(_runs)}" if _runs else "")
    with st.container(key="lines"):
        if not _runs:
            st.caption("None yet.")
        for r in _runs[:40]:
            when = datetime.fromtimestamp(r["mtime"]).strftime("%d-%b %H:%M")
            color = LABEL_COLOR.get(r["label"], "gray")
            st.button(f"`{when}` {md_safe(r['title'])} · :{color}[{md_safe(r['label'])}]", key=f"run_{r['id']}",
                      type="tertiary", on_click=go, kwargs=dict(run=r["id"]))
    with st.expander("Session"):
        st.caption(f"API key: {_key}")
        _k = st.text_input("Set API key (this session only)", type="password", key="api_key_input")
        if _k:
            os.environ["ANTHROPIC_API_KEY"] = _k      # process environment only; never written to disk
        if (RUNS_DIR / "index.html").exists():
            st.button("Compare runs", key="compare_btn", type="tertiary", on_click=go, kwargs=dict(compare=True))
        if (RUNS_DIR / "mock_app" / "progress.json").exists():
            st.button("Open example run (mock)", key="mock_btn", type="tertiary", on_click=go,
                      kwargs=dict(run="mock_app"))


# ----------------------------------------------------------------------------
# main area pieces
# ----------------------------------------------------------------------------
def render_preview(slug: str) -> None:
    label, rel, _ = next(r for r in REGISTRY if r[2] == slug)
    authors, year, topic = paper_parts(label)
    st.markdown(f'<div class="runh"><span class="t">{esc(label)}</span></div>', unsafe_allow_html=True)
    p = abs_path(rel)
    if not p.exists():
        st.caption(f"File not found: {rel}")
        return
    try:
        n, title, text = pdf_preview(str(p), p.stat().st_mtime)
        st.caption(f"{title + ' · ' if title else ''}{n} pages")
        st.text(text or "(no extractable text on page 1)")
    except Exception as e:  # noqa: BLE001
        st.caption(f"Preview unavailable: {type(e).__name__}")
    if slug in ss.queue:
        st.caption("Selected. Press Run in the right column.")
    else:
        st.button("Use this paper", key=f"use_{slug}", on_click=toggle_paper, args=(slug,))


def render_steps(p: dict | None) -> None:
    stages = p["stages"] if p else [{"n": n, "name": nm, "status": "pending", "detail": ""} for n, nm in H.STAGE_NAMES]
    rows = []
    for s in stages:
        detail = esc(s["detail"]) if s["status"] != "pending" else ""
        rows.append(f'<div class="step s-{s["status"]}"><span class="n">{s["n"]}</span>'
                    f'<span class="name">{esc(s["name"])}</span><span class="detail">{detail}</span></div>')
    st.markdown(f'<div class="steps">{"".join(rows)}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cumulative(path: str, mtime: float, start: str, end: str) -> pd.DataFrame | None:
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.loc[(df.index >= start) & (df.index <= end)]
    cols = [c for c in ("strategy_net", "spy", "ew_universe") if c in df.columns]
    if not cols or df.empty:
        return None
    out = ((1 + df[cols].fillna(0)).cumprod() - 1) * 100
    out = out.resample("W-FRI").last().dropna(how="all")      # weekly points: the axis then labels years
    return out.rename(columns={"strategy_net": "strategy (net)", "spy": "SPY", "ew_universe": "EW universe"})


def render_quote(run_dir: Path, s: dict) -> None:
    perf = s.get("performance") or {}
    net, ew, spy = perf.get("strategy_net") or {}, perf.get("ew_universe") or {}, perf.get("spy") or {}
    al = s.get("alpha") or {}
    rec = (H.read_progress(run_dir) or {}).get("recommendation") or ""
    cls = {"green": "up", "orange": "amber", "red": "dn"}.get(LABEL_COLOR.get(rec, "gray"), "mute")
    d = net.get("cagr", float("nan")) - ew.get("cagr", float("nan"))
    arrow = "▲" if d >= 0 else "▼"
    st.markdown(f"""<div class="quote">
<span><span class="k">Net CAGR</span><span class="v">{H.pct(net.get("cagr"))}</span> <span class="{"up" if d >= 0 else "dn"}">{arrow} {d:+.1%} vs EW</span></span>
<span><span class="k">EW universe</span><span class="v">{H.pct(ew.get("cagr"))}</span></span>
<span><span class="k">SPY</span><span class="v">{H.pct(spy.get("cagr"))}</span></span>
<span><span class="k">Sharpe</span><span class="v">{H.num(net.get("sharpe"))}</span></span>
<span><span class="k">Max DD</span><span class="v">{H.pct(net.get("max_drawdown"))}</span></span>
<span><span class="k">Alpha t</span><span class="v">{H.num(al.get("alpha_t"))}</span></span>
<span><span class="k">Verdict</span><span class="v {cls}">{esc(rec) or "—"}</span></span>
</div>""", unsafe_allow_html=True)


def render_chart(run_dir: Path, start: str, end: str) -> None:
    f = run_dir / "returns.parquet"
    if not f.exists():
        return
    try:
        cum = cumulative(str(f), f.stat().st_mtime, start, end)
    except Exception as e:  # noqa: BLE001
        st.caption(f"Returns unreadable: {type(e).__name__}")
        return
    if cum is None:
        return
    panel("Cumulative return, %", f"{start} → {end}")
    import altair as alt
    cum.index.name = "date"
    long = cum.reset_index().melt("date", var_name="series", value_name="pct")
    chart = (alt.Chart(long).mark_line(strokeWidth=1.4).encode(
        x=alt.X("date:T", axis=alt.Axis(format="%Y", tickCount="year", grid=False, title=None, labelColor="#9a9a9a")),
        y=alt.Y("pct:Q", axis=alt.Axis(title=None, format="d", gridColor="#262626", labelColor="#9a9a9a", domain=False)),
        color=alt.Color("series:N", scale=alt.Scale(domain=list(cum.columns), range=[ORANGE, GREY, PURPLE][:cum.shape[1]]),
                        legend=alt.Legend(title=None, orient="top-left", labelColor="#d9d9d9")),
        tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), "series:N", alt.Tooltip("pct:Q", format=".1f")])
        .properties(height=230).configure_view(stroke=None).configure(background="transparent"))
    st.altair_chart(chart, width="stretch")


def sections(obj, heading_level: str = "**") -> None:
    for title, df in H.table_sections(obj):
        if title:
            st.markdown(f"{heading_level}{title}{heading_level}")
        st.dataframe(df, width="stretch")


def tab_extraction(run_dir: Path) -> None:
    d = H.read_json(run_dir / "spec_consensus.json")
    if not isinstance(d, dict):
        st.caption(NA)
        return
    spec = d.get("spec", d)
    val = H.read_json(run_dir / "validation.json")
    if spec.get("can_implement", True):
        st.markdown("**Implementable** with the data in this snapshot.")
    else:
        st.markdown("**Not implementable**")
        for r in spec.get("cannot_implement_reasons", []):
            st.markdown(f"- {md_safe(r)}")
    st.caption(f"{d.get('n_runs', '?')} extraction run(s); selected experiment: "
               f"{spec.get('selected_experiment', 'not stated')}")
    rows, quotes = [], []
    for f in MATERIAL_FIELDS:
        if f not in spec:
            continue
        value, page, quote = H.field_value_and_evidence(spec, f)
        ok = H.evidence_ok(val, f)
        rows.append({"Field": H.pretty(f), "Value": value, "Page": "" if page is None else str(page),
                     "Evidence": "" if ok is None else ("yes" if ok else "no"),
                     "Agreement": H.agreement_text(d.get("agreement"), f)})
        quotes.append((f, page, quote))
    if rows:
        st.table(pd.DataFrame(rows).set_index("Field"))       # st.table wraps long values; st.dataframe clips them
        st.markdown("**Evidence quotes**")
        for f, page, quote in quotes:
            with st.expander(f"{H.pretty(f)}" + (f" · p. {page}" if page is not None else "")):
                st.write(quote or "no quote recorded")
    un = spec.get("unresolved_details") or []
    st.markdown("**Unresolved details**")
    if un:
        for u in un:
            st.markdown(f"- {md_safe(u)}")
    else:
        st.caption("None recorded.")
    if isinstance(val, dict) and "n_check_disagreements" in val:
        st.caption(f"Second-pass check disagreements: {val['n_check_disagreements']}")
    ref = H.read_json(run_dir / "scorecard_interpretation.json")
    if isinstance(ref, dict) and ref.get("n_compared"):
        with st.expander("Comparison with the reference extraction"):
            st.write(f"{ref.get('n_compared')} fields compared; agreement {H.pct(ref.get('agreement_rate'), 0)}. "
                     f"Critical differences: {', '.join(map(H.pretty, ref.get('critical_errors') or [])) or 'none'}. "
                     f"Minor differences: {', '.join(map(H.pretty, ref.get('minor_errors') or [])) or 'none'}.")


def table(title: str, df: pd.DataFrame | None, note: str = "") -> None:
    if df is None or df.empty:
        return
    st.markdown(f"**{title}**")
    if note:
        st.caption(note)
    st.dataframe(df, width="stretch", hide_index=df.index.equals(pd.RangeIndex(len(df))))


def safely(fn, *args) -> None:
    """One malformed panel must never take the tab down."""
    try:
        fn(*args)
    except Exception as ex:  # noqa: BLE001
        st.caption(f"Panel unavailable: {type(ex).__name__}")


def tab_evaluation(run_dir: Path) -> None:
    s = H.read_json(run_dir / "scorecard_strategy.json")
    if not isinstance(s, dict):
        st.caption(NA)
        return
    m = s.get("meta") if isinstance(s.get("meta"), dict) else {}
    st.caption(f"Daily returns {str(m.get('first_date', '?'))[:10]} to {str(m.get('last_date', '?'))[:10]} "
               f"({m.get('n_days', '?')} days); base trading cost {m.get('cost_bps_base', '?')} bps; "
               f"beta to SPY {H.num(s.get('beta_vs_spy'))}.")
    safely(table, "Performance", H.performance_table(s))
    safely(table, "Relative to the benchmarks", H.relative_table(s),
           "Monthly differences, Newey-West standard errors.")
    safely(table, "Alpha and spanning", H.alpha_table(s), "Daily regressions on market, size, value and momentum factors.")
    st.markdown("#### Robustness")
    safely(table, "Trading cost grid", H.cost_grid_table(s))
    safely(table, "Sub-periods", H.subperiod_table(s))

    def groups() -> None:
        df, cap = H.groups_table(s)
        table("Ranked groups", df, cap)
    safely(groups)
    safely(table, "Implementation", H.implementation_table(s))
    if isinstance(s.get("definitions"), dict):
        with st.expander("Definitions"):
            for k, v in s["definitions"].items():
                st.markdown(f"**{H.pretty(k)}.** {md_safe(v)}")


def tab_economics(run_dir: Path) -> None:
    e = H.read_json(run_dir / "economics.json")
    mc_path = run_dir / "montecarlo.parquet"
    if not isinstance(e, dict) and not mc_path.exists():
        st.caption(NA)
        return
    if isinstance(e, dict):
        st.caption(f"{e.get('n_paths', '?')} bootstrapped paths, {e.get('horizon_days', '?')}-day horizon, "
                   f"{e.get('mean_block_days', '?')}-day mean block; benchmark {H.pretty(e.get('benchmark', '?'))}.")
        safely(table, "Share of paths", H.fractions_table(e))
        safely(table, "Fund economics", H.money_table(e),
               "Incremental fees are measured against an equal-weight universe fund under the same fee schedule.")
        safely(table, "Distribution of outcomes", H.quantiles_table(e), "Percentiles across paths.")
        safely(table, "Sensitivity to bootstrap block length", H.block_table(e))
    if mc_path.exists():
        try:
            mc = pd.read_parquet(mc_path)
            for cols, name in ((["npv", "npv_demeaned"], "NPV"),
                               (["end_wealth_strategy", "end_wealth_ew", "end_wealth_spy"], "Ending wealth")):
                h = H.histograms(mc, "", cols=cols)
                if h is not None:
                    unit = f", USD {h.attrs['unit']}" if h.attrs.get("unit") else ""
                    st.markdown(f"**{name}** (histogram over {len(mc):,} paths{unit})")
                    st.bar_chart(h, x="bin", y=[c for c in h.columns if c != "bin"],
                                 color=_series_colors(len(h.columns) - 1), stack=False)
        except Exception as ex:  # noqa: BLE001
            st.caption(f"Monte Carlo paths unreadable: {type(ex).__name__}")
    if isinstance(e, dict) and e.get("assumptions"):
        with st.expander("Assumptions"):
            for a in e["assumptions"]:
                st.markdown(f"- {md_safe(a)}")


def _series_colors(n: int) -> list[str]:
    return [ORANGE, GREY, PURPLE, "#5c5c5c"][:max(n, 1)]


def tab_dossier(run_dir: Path, busy: bool) -> None:
    f = run_dir / "dossier.html"
    meta = H.read_json(run_dir / "paper_meta.json") or {}
    c1, c2, _ = st.columns([2, 2, 5])
    if f.exists():
        c1.download_button("Download dossier", f.read_bytes(), file_name=f"{run_dir.name}_dossier.html",
                           mime="text/html", key=f"dl_{run_dir.name}")
    can_regen = bool(meta.get("source_path")) and (run_dir / "scorecard_strategy.json").exists()
    if can_regen and c2.button("Regenerate narrative", key=f"regen_{run_dir.name}", disabled=busy):
        _, _, model, _ = run_window(run_dir)
        start_job([dict(label=run_dir.name, pdf=meta["source_path"], experiment=meta.get("selected_experiment"),
                        prefix=run_dir.name[:12], run_dir=str(run_dir), start_stage=9, model=model)])
        go(run=run_dir.name)
        st.rerun()
    if f.exists():
        st.iframe(f, height=1200)
    else:
        st.caption(NA)


def render_summary(p: dict | None, state: str) -> None:
    panel("Summary")
    if p is None:
        st.write("Select a paper or drop a PDF.")
        return
    if state == "blocked" or (state == "failed" and not p.get("recommendation")):
        st.markdown("**Unable to evaluate.**" if state == "blocked" else "**Run failed.**")
        reasons = [s["detail"] for s in p["stages"] if s["status"] in ("blocked", "failed") and s["detail"]]
        text = "; ".join(reasons) or p.get("error") or (p.get("summary") or "").removeprefix("Unable to evaluate.").strip()
        st.write(md_safe(text))
    elif state == "finished":
        st.markdown(f"**{md_safe(p.get('recommendation') or 'no recommendation')}**")
        st.write(md_safe(p.get("summary") or ""))
    elif state == "running":
        st.write("Running…")
    else:
        st.write("Not finished.")


# ----------------------------------------------------------------------------
# main area
# ----------------------------------------------------------------------------
@st.fragment(run_every=1 if active else None)
def main_view() -> None:
    job_now: Job | None = holder()["job"]
    is_active = bool(job_now and job_now.active)
    if ss.get("_was_active") and not is_active:     # job just finished: refresh the right column and polling
        ss["_was_active"] = False
        st.rerun()
    ss["_was_active"] = is_active

    if ss.compare:
        panel("Compare runs")
        idx = RUNS_DIR / "index.html"
        if idx.exists():
            st.iframe(idx, height=900)
        else:
            st.caption(NA)
        return

    run_dir: Path | None = None
    if ss.selected_run == FOLLOW:
        if job_now:
            run_dir = job_now.current_dir() or job_now.last_dir
    elif ss.selected_run:
        run_dir = RUNS_DIR / ss.selected_run

    if run_dir is None and ss.preview:
        render_preview(ss.preview)
        render_summary(None, "missing")
        return

    if run_dir is None and ss.selected_run != FOLLOW:
        panel("Overview")
        st.write("Select a paper tile on the right or drop a PDF, then press Run. "
                 "Previous runs open from the list below the parameters.")
        render_summary(None, "missing")
        return

    p = H.read_progress(run_dir) if run_dir else None
    meta = (H.read_json(run_dir / "paper_meta.json") if run_dir else None) or {}
    title = (p or {}).get("paper_title") or meta.get("title") or (run_dir.name if run_dir else "Starting…")
    slug = (p or {}).get("strategy_slug")
    ws, we, model, custom = run_window(run_dir) if run_dir else (_ws, _we, _sel_model, False)
    st.markdown(f'<div class="runh"><span class="t">{esc(title)}</span>'
                f'<span class="mute">strategy: {esc(slug) if slug else "not yet determined"}</span>'
                f'<span class="win">BACKTEST {ws} → {we}{" · custom" if custom else ""}</span>'
                f'<span class="mute">{esc(model)}</span></div>', unsafe_allow_html=True)
    sc = H.read_json(run_dir / "scorecard_strategy.json") if run_dir else None
    if isinstance(sc, dict):
        safely(render_quote, run_dir, sc)
    panel("Pipeline")
    render_steps(p)
    if job_now and job_now.errors:
        for e in job_now.errors:
            st.markdown(f'<div class="err">{esc(e)}</div>', unsafe_allow_html=True)

    if run_dir is not None:
        render_chart(run_dir, ws, we)
        t1, t2, t3, t4 = st.tabs(["Extraction", "Evaluation", "Economics", "Dossier"])
        with t1:
            tab_extraction(run_dir)
        with t2:
            tab_evaluation(run_dir)
        with t3:
            tab_economics(run_dir)
        with t4:
            tab_dossier(run_dir, is_active)
    mine = is_active and run_dir is not None and job_now.current_dir() == run_dir
    state = H.run_state(p, job_active=mine) if p else ("running" if is_active else "missing")
    if p is None and not is_active:
        st.caption(NA)
    render_summary(p if p else ({"stages": []} if is_active else None), state)


with left:
    main_view()

st.markdown(f"""<div class="foot"><b>Papers:</b> click one tile, then press Run; the nine papers are already on disk, nothing
needs uploading. <b>Dropped PDFs:</b> the paper is read and the strategy extracted for any PDF, but the engine only backtests
long-only monthly sorts on daily US equity prices ({_manifest.get("n_universe", "?")} S&amp;P 500 names, {WARMUP_START[:4]}–{EVAL_END[:4]}).
If the extracted strategy needs other data (fundamentals, futures, options, intraday) or another construction, the run stops
after extraction and validation with the reason, and only the Extraction tab and the blocked dossier are produced.
<b>Model steps</b> (extraction, check pass, narrative) need an Anthropic key in <code>.env</code> unless a cached answer exists.</div>""",
            unsafe_allow_html=True)
