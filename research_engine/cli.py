"""Command line: engine ingest|extract|check|run|report|all|app"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

from . import contracts as C


def main(argv=None):
    ap = argparse.ArgumentParser(prog="engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("all", help="run the full pipeline on one PDF"); a.add_argument("pdf"); a.add_argument("--select"); a.add_argument("--run-dir"); a.add_argument("--llm", default="api", choices=["api", "files_only"]); a.add_argument("--n", type=int, default=3); a.add_argument("--reference"); a.add_argument("--from-stage", type=int, default=1); a.add_argument("--model"); a.add_argument("--start", help="backtest start YYYY-MM-DD"); a.add_argument("--end", help="backtest end YYYY-MM-DD")
    r = sub.add_parser("resume", help="resume a run folder from a stage"); r.add_argument("run_dir"); r.add_argument("--from-stage", type=int, required=True); r.add_argument("--llm", default="api")
    sub.add_parser("index", help="rebuild runs/index.html")
    e = sub.add_parser("export-prompts", help="write session prompts for a run"); e.add_argument("run_dir"); e.add_argument("--k", type=int, default=1)
    sub.add_parser("app", help="launch the Streamlit app")
    ns = ap.parse_args(argv)
    if ns.cmd == "all":
        from .pipeline import run_pipeline
        rd = run_pipeline(Path(ns.pdf), Path(ns.run_dir) if ns.run_dir else None, ns.select, llm=ns.llm,
                          n_extractions=ns.n, reference_spec=Path(ns.reference) if ns.reference else None,
                          start_stage=ns.from_stage, model=ns.model, eval_start=ns.start, eval_end=ns.end)
        print(rd)
    elif ns.cmd == "resume":
        from .pipeline import run_pipeline
        import json
        meta = json.loads((Path(ns.run_dir) / C.RunFiles.paper_meta).read_text())
        run_pipeline(Path(meta["source_path"]), Path(ns.run_dir), meta.get("selected_experiment"), llm=ns.llm, start_stage=ns.from_stage)
    elif ns.cmd == "index":
        from .reporting.render import render_index
        render_index(C.RUNS_DIR)
    elif ns.cmd == "export-prompts":
        from .extraction.session_runner import export_prompt
        print(export_prompt(Path(ns.run_dir), ns.k))
    elif ns.cmd == "app":
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).parent / "app" / "streamlit_app.py")])


if __name__ == "__main__":
    main()
