"""Dropped PDFs run extraction only: read, extract, verdict, strategy summary; no backtest."""
import json
import shutil
from pathlib import Path

import pytest

from research_engine.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "runs" / "demo_max"
PDF = ROOT / "papers" / "pdf" / "bali_cakici_whitelaw_2011_maxing_out_nber14804.pdf"


@pytest.mark.skipif(not (DEMO / "spec_raw_1.json").exists() or not PDF.exists(), reason="demo_max or its PDF missing")
def test_extract_only_skips_the_engine_and_writes_a_summary(tmp_path):
    run = tmp_path / "x"
    run.mkdir()
    for f in DEMO.glob("spec_raw_*.json"):
        shutil.copy(f, run / f.name)
    run_pipeline(PDF, run, llm="files_only", extract_only=True)
    p = json.loads((run / "progress.json").read_text())
    st = {s["n"]: s["status"] for s in p["stages"]}
    assert st[1] == st[2] == st[3] == "done"
    assert all(st[n] == "skipped" for n in range(4, 9))
    assert st[9] == "done" and p["stages"][8]["name"] == "Strategy summary"
    assert p["recommendation"] == "extraction only"
    summ = json.loads((run / "strategy_summary.json").read_text())
    assert summ["source"] == "template" and summ["summary"] and summ["implementation_note"]
    assert p["summary"] == summ["summary"]
    assert "engine could run it as 'max'" in p["stages"][2]["detail"]
    assert not (run / "returns.parquet").exists() and not (run / "dossier.html").exists()
    assert json.loads((run / "run_config.json").read_text())["extract_only"] is True
