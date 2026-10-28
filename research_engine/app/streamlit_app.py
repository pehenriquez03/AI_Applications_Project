"""Local Streamlit front end for the research-screening engine.

Launch:  .venv/bin/streamlit run research_engine/app/streamlit_app.py
Reads run folders under runs/ (progress.json is polled while a run is active);
runs the pipeline in a background thread. Never writes the API key to disk.
"""
from __future__ import annotations

import hashlib
import html
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from research_engine.app import helpers as H
from research_engine.app.papers_registry import REGISTRY, abs_path
from research_engine.contracts import MATERIAL_FIELDS, MODEL, RUNS_DIR, SNAPSHOT_DIR

ACCENT = "#3b6ea5"
FOLLOW = "__job__"          # selected_run sentinel: follow whichever run the job is on
NA = "not available yet"

st.set_page_config(page_title="Research Screening Agent", layout="wide")

st.markdown(f"""<style>
.stApp, .stApp *:not([data-testid="stIconMaterial"]) {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
.stApp code, .stApp pre, .stApp pre * {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
.stApp h1 {{ font-size: 1.8rem; font-weight: 600; padding-bottom: .1rem; }}
.stApp h2, .stApp h3, .stApp h4 {{ font-weight: 600; }}
section[data-testid="stSidebar"] .stButton button {{ justify-content: flex-start; text-align: left; padding: .15rem .25rem; }}
section[data-testid="stSidebar"] .stButton button p {{ text-align: left; font-size: .9rem; }}
.sub {{ color: #8a8a8a; margin-top: -.6rem; font-size: .95rem; }}
.steps {{ margin: .4rem 0 1.2rem; font-variant-numeric: tabular-nums; }}
.step {{ display: flex; gap: 1rem; line-height: 1.55; }}
.step .n {{ width: 1.2rem; }}
.step .name {{ width: 12rem; flex: none; }}
.step .detail {{ flex: 1; }}
.s-done {{ color: #8a8a8a; }}
.s-running {{ color: {ACCENT}; }}
.s-pending {{ color: #b3b3b3; }}
.s-blocked, .s-failed {{ color: #a65d5d; }}
.stApp hr {{ border-color: rgba(128,128,128,.25); margin: 1.5rem 0 .5rem; }}
.stApp [data-testid="stSidebar"] .side-h {{ font-weight: 600; margin: .8rem 0 .2rem; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {ACCENT}; }}
.stTabs [aria-selected="true"], .stTabs [aria-selected="true"] p {{ color: {ACCENT}; }}
.stApp button[kind="primary"] {{ background-color: {ACCENT}; border-color: {ACCENT}; color: #fff; }}
.err {{ color: #a65d5d; font-size: .9rem; }}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# background job (one at a time; survives browser refresh via cache_resource)
# ----------------------------------------------------------------------------
@st.cache_resource
def holder() -> dict:
    return {"job": None}


class Job:
    """tasks: dicts with label, pdf, experiment, prefix (sha12), run_dir, start_stage."""

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
        try:
            d = run_pipeline(Path(t["pdf"]), run_dir=t.get("run_dir"),
                             selected_experiment=t.get("experiment") or None,
                             llm="api", start_stage=t.get("start_stage", 1))
            job.last_dir = Path(d)
        except BaseException as e:  # noqa: BLE001 - recorded, next paper still runs
            job.errors.append(f"{t['label']}: {type(e).__name__}: {e}")
            job.last_dir = job.current_dir() or job.last_dir
    job.active = False


def start_job(tasks: list[dict]) -> None:
    job = Job(tasks)
    holder()["job"] = job
    threading.Thread(target=_work, args=(job,), daemon=True).start()


# ----------------------------------------------------------------------------
# state
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


# ----------------------------------------------------------------------------
# header
# ----------------------------------------------------------------------------
st.title("Research Screening Agent")
try:
    _m = H.read_json(SNAPSHOT_DIR / "manifest.json") or {}
    st.markdown(f'<div class="sub">{esc(MODEL)} · {_m.get("n_universe", "?")} S&amp;P 500 names, 2015–2025 · '
                f'snapshot {esc(_m.get("snapshot", "?"))}</div>', unsafe_allow_html=True)
except Exception:
    pass

# ----------------------------------------------------------------------------
# left column
# ----------------------------------------------------------------------------
sb = st.sidebar
sb.markdown('<div class="side-h">Papers</div>', unsafe_allow_html=True)
for label, rel, slug in REGISTRY:
    sb.button(label, key=f"paper_{slug}", type="tertiary", on_click=go, kwargs=dict(preview=slug))
if ss.queue:
    names = {s: l for l, _, s in REGISTRY}
    sb.caption("Queued: " + "; ".join(names[s].split(" — ")[0] for s in ss.queue))
    sb.button("Clear queue", key="clear_q", type="tertiary", on_click=lambda: ss.queue.clear())

sb.markdown('<div class="side-h">Drop a PDF</div>', unsafe_allow_html=True)
uploads = sb.file_uploader("PDF files to screen", type=["pdf"], accept_multiple_files=True,
                           key=f"uploader_{ss.upload_n}", label_visibility="collapsed")
experiment = sb.text_input("Experiment (optional)", key="experiment",
                           placeholder="blank = the paper's main experiment")

if sb.button("Run", type="primary", disabled=active or not (ss.queue or uploads), key="run_btn"):
    tasks, problems = [], []
    for slug in ss.queue:
        label, rel, _ = next(r for r in REGISTRY if r[2] == slug)
        p = abs_path(rel)
        if not p.exists():
            problems.append(f"missing file: {rel}")
            continue
        tasks.append(dict(label=label, pdf=str(p), experiment=experiment,
                          prefix=hashlib.sha256(p.read_bytes()).hexdigest()[:12]))
    for f in uploads or []:
        data = f.getvalue()
        sha = hashlib.sha256(data).hexdigest()
        dest = RUNS_DIR / "_uploads" / f"{sha}.pdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(data)
        tasks.append(dict(label=f.name, pdf=str(dest), experiment=experiment, prefix=sha[:12]))
    for pr in problems:
        sb.error(pr)
    if tasks:
        start_job(tasks)
        ss.queue = []
        ss.upload_n += 1
        go(run=FOLLOW)
        st.rerun()

sb.markdown('<div class="side-h">Previous runs</div>', unsafe_allow_html=True)
_runs = H.list_runs(RUNS_DIR)
if not _runs:
    sb.caption("None yet.")
for r in _runs:
    sb.button(f"{r['title']} · {r['label']}", key=f"run_{r['id']}", type="tertiary",
              on_click=go, kwargs=dict(run=r["id"]))

with sb.expander("Settings"):
    st.text_input("Model", MODEL, disabled=True)
    st.caption(f"API key: {H.api_key_status(ROOT, os.environ)}")
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
@st.cache_data(show_spinner=False)
def pdf_preview(path: str) -> tuple[int, str]:
    from pypdf import PdfReader
    r = PdfReader(path)
    text = (r.pages[0].extract_text() or "") if len(r.pages) else ""
    return len(r.pages), " ".join(text.split())[:1500]


def render_preview(slug: str) -> None:
    label, rel, _ = next(r for r in REGISTRY if r[2] == slug)
    st.markdown(f"### {label}")
    p = abs_path(rel)
    if not p.exists():
        st.caption(f"File not found: {rel}")
        return
    try:
        n, text = pdf_preview(str(p))
        st.caption(f"{n} pages")
        st.text(text or "(no extractable text on page 1)")
    except Exception as e:  # noqa: BLE001
        st.caption(f"Preview unavailable: {type(e).__name__}")
    if slug in ss.queue:
        st.caption("Queued. Press Run in the left column.")
    else:
        st.button("Use this paper", key=f"use_{slug}", on_click=lambda: ss.queue.append(slug))


def render_steps(p: dict | None) -> None:
    stages = p["stages"] if p else [{"n": n, "name": nm, "status": "pending", "detail": ""} for n, nm in H.STAGE_NAMES]
    rows = []
    for s in stages:
        detail = esc(s["detail"]) if s["status"] != "pending" else ""
        rows.append(f'<div class="step s-{s["status"]}"><span class="n">{s["n"]}</span>'
                    f'<span class="name">{esc(s["name"])}</span><span class="detail">{detail}</span></div>')
    st.markdown(f'<div class="steps">{"".join(rows)}</div>', unsafe_allow_html=True)


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
            st.markdown(f"- {r}")
    st.caption(f"{d.get('n_runs', '?')} extraction run(s); selected experiment: "
               f"{spec.get('selected_experiment', 'not stated')}")
    rows, quotes = [], []
    for f in MATERIAL_FIELDS:
        if f not in spec:
            continue
        value, page, quote = H.field_value_and_evidence(spec, f)
        ok = H.evidence_ok(val, f)
        rows.append({"Field": H.pretty(f), "Value": value, "Page": "" if page is None else str(page),
                     "Evidence": "" if ok is None else ("✓" if ok else "✗"),
                     "Agreement": H.agreement_text(d.get("agreement"), f)})
        quotes.append((f, page, quote))
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown("**Evidence quotes**")
        for f, page, quote in quotes:
            with st.expander(f"{H.pretty(f)}" + (f" · p. {page}" if page is not None else "")):
                st.write(quote or "no quote recorded")
    un = spec.get("unresolved_details") or []
    st.markdown("**Unresolved details**")
    if un:
        for u in un:
            st.markdown(f"- {u}")
    else:
        st.caption("None recorded.")
    if isinstance(val, dict) and "n_check_disagreements" in val:
        st.caption(f"Second-pass check disagreements: {val['n_check_disagreements']}")


def tab_evaluation(run_dir: Path) -> None:
    s = H.read_json(run_dir / "scorecard_strategy.json")
    i = H.read_json(run_dir / "scorecard_interpretation.json")
    if s is None and i is None:
        st.caption(NA)
        return
    st.markdown("#### Strategy")
    if s is not None:
        sections(s)
    else:
        st.caption(NA)
    st.markdown("#### Interpretation")
    if i is not None:
        sections(i)
    else:
        st.caption(NA)


def tab_economics(run_dir: Path) -> None:
    e = H.read_json(run_dir / "economics.json")
    mc_path = run_dir / "montecarlo.parquet"
    if e is None and not mc_path.exists():
        st.caption(NA)
        return
    if e is not None:
        sections(e)
    if mc_path.exists():
        try:
            mc = pd.read_parquet(mc_path)
            for needle, name in (("npv", "NPV of the fee stream"), ("wealth", "Ending wealth")):
                h = H.histograms(mc, needle)
                if h is not None:
                    unit = f", {h.attrs['unit']}" if h.attrs.get("unit") else ""
                    st.markdown(f"**{name}** (distribution over {len(mc):,} paths{unit})")
                    st.bar_chart(h, x="bin", y=[c for c in h.columns if c != "bin"])
        except Exception as ex:  # noqa: BLE001
            st.caption(f"Monte Carlo paths unreadable: {type(ex).__name__}")


def tab_dossier(run_dir: Path, busy: bool) -> None:
    f = run_dir / "dossier.html"
    if not f.exists():
        st.caption(NA)
        return
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("Download dossier", f.read_bytes(), file_name=f"{run_dir.name}_dossier.html",
                       mime="text/html", key=f"dl_{run_dir.name}")
    if c2.button("Regenerate narrative", key=f"regen_{run_dir.name}", disabled=busy):
        meta = H.read_json(run_dir / "paper_meta.json") or {}
        pdf = meta.get("source") or meta.get("source_path") or meta.get("pdf_path") or ""
        (run_dir / "narrative.json").unlink(missing_ok=True)
        start_job([dict(label=run_dir.name, pdf=pdf, experiment=meta.get("selected_experiment"),
                        prefix=run_dir.name[:12], run_dir=str(run_dir), start_stage=9)])
        go(run=run_dir.name)
        st.rerun()
    components.html(f.read_text(errors="replace"), height=1200, scrolling=True)


def render_summary(p: dict | None, state: str) -> None:
    st.divider()
    st.subheader("Summary")
    if p is None:
        st.write("Select a paper or drop a PDF.")
        return
    if state == "blocked" or (state == "failed" and not p.get("recommendation")):
        st.markdown("**Unable to evaluate.**" if state == "blocked" else "**Run failed.**")
        reasons = [s["detail"] for s in p["stages"] if s["status"] in ("blocked", "failed") and s["detail"]]
        text = "; ".join(reasons) or p.get("error") or (p.get("summary") or "").removeprefix("Unable to evaluate.").strip()
        st.write(text.replace("$", "\\$"))
    elif state == "finished":
        st.markdown(f"**{(p.get('recommendation') or 'no recommendation').replace('$', chr(92) + '$')}**")
        st.write((p.get("summary") or "").replace("$", "\\$"))
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
    if ss.get("_was_active") and not is_active:     # job just finished: refresh sidebar and polling
        ss["_was_active"] = False
        st.rerun()
    ss["_was_active"] = is_active

    if ss.compare:
        idx = RUNS_DIR / "index.html"
        if idx.exists():
            components.html(idx.read_text(errors="replace"), height=900, scrolling=True)
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
        render_summary(None, "missing")
        return

    p = H.read_progress(run_dir) if run_dir else None
    meta = (H.read_json(run_dir / "paper_meta.json") if run_dir else None) or {}
    title = (p or {}).get("paper_title") or meta.get("title") or (run_dir.name if run_dir else "Starting…")
    st.markdown(f"### {esc(title)}")
    slug = (p or {}).get("strategy_slug")
    st.caption(f"strategy: {slug}" if slug else "strategy: not yet determined")
    render_steps(p)
    if job_now and job_now.errors:
        for e in job_now.errors:
            st.markdown(f'<div class="err">{esc(e)}</div>', unsafe_allow_html=True)

    if run_dir is not None:
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


main_view()
