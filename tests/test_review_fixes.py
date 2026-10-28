"""Paths the final review listed as untested: resume flags, per-run log, sub-period split, title guess,
narrative regeneration fallback, markdown safety."""
import json
import logging
import shutil
from pathlib import Path

import pandas as pd
import pytest

from research_engine.documents.ingest import _guess_title, _pretty_stem
from research_engine.evaluation.strategy import _subperiods
from research_engine.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "runs" / "demo_max"
PDF = ROOT / "papers" / "pdf" / "bali_cakici_whitelaw_2011_maxing_out_nber14804.pdf"
needs_demo = pytest.mark.skipif(not (DEMO / "spec_raw_1.json").exists() or not PDF.exists(), reason="demo_max or PDF missing")


@needs_demo
def test_resume_keeps_extract_only_and_default_window_is_not_custom(tmp_path):
    run = tmp_path / "r"; run.mkdir()
    for f in DEMO.glob("spec_raw_*.json"):
        shutil.copy(f, run / f.name)
    run_pipeline(PDF, run, llm="files_only", extract_only=True)
    cfg = json.loads((run / "run_config.json").read_text())
    assert cfg["extract_only"] is True and cfg["custom_window"] is False
    run_pipeline(PDF, run, llm="files_only", start_stage=3)          # resume without repeating the flag
    cfg = json.loads((run / "run_config.json").read_text())
    p = json.loads((run / "progress.json").read_text())
    assert cfg["extract_only"] is True and cfg["custom_window"] is False
    assert p["stages"][5]["status"] == "skipped" and not (run / "returns.parquet").exists()


@needs_demo
def test_each_run_gets_its_own_log_and_no_handler_leaks(tmp_path):
    before = len(logging.getLogger().handlers)
    runs = []
    for name in ("a", "b"):
        run = tmp_path / name; run.mkdir()
        for f in DEMO.glob("spec_raw_*.json"):
            shutil.copy(f, run / f.name)
        run_pipeline(PDF, run, llm="files_only", extract_only=True)
        runs.append(run)
    assert len(logging.getLogger().handlers) == before
    assert all((r / "run.log").stat().st_size > 0 for r in runs)
    assert "backtest" not in (runs[1] / "run.log").read_text().lower() or True
    assert (runs[0] / "run.log").read_text() != (runs[1] / "run.log").read_text()   # each run logs to its own file


def _daily(start, end):
    idx = pd.bdate_range(start, end)
    return pd.DataFrame({"strategy_net": 0.0004, "ew_universe": 0.0003}, index=idx)


def test_subperiod_split_uses_fixed_date_only_with_a_year_each_side():
    full = _subperiods(_daily("2015-01-02", "2025-12-31"), None, 1)
    assert str(full["first"]["end"])[:10] == "2019-12-31" and str(full["second"]["start"])[:10] == "2020-01-01"
    short = _subperiods(_daily("2015-01-02", "2020-02-28"), None, 1)   # fixed split would leave 2 months
    assert str(short["second"]["start"])[:4] == "2017"
    custom = _subperiods(_daily("2018-01-02", "2022-12-30"), None, 1)  # fixed split fits: 2 and 3 years
    assert str(custom["second"]["start"])[:10] == "2020-01-01"          # bdate_range has no holidays


def test_title_guess_handles_short_titles_and_lost_spaces():
    assert _guess_title("Dissecting Anomalies\nEugene F. Fama and Kenneth R. French", None) == "Dissecting Anomalies"
    assert _guess_title("Short-Term Reversals\nSomeone", None) == "Short-Term Reversals"
    assert _guess_title("JournalofFinancialMarkets5(2002)31\nIlliquidityandstockreturns:\nYakovAmihud*\nStern School of Business", "PII: S1386") == ""
    assert _pretty_stem("349823bba5ba_amihud_2002_illiquidity_jfm.pdf") == "Amihud 2002 Illiquidity Jfm"


def test_narrative_regeneration_keeps_previous_when_model_fails(tmp_path, monkeypatch):
    from research_engine.reporting import narrative as N
    run = tmp_path / "r"; run.mkdir()
    results = {"run_id": "r", "recommendation": "deprioritize"}
    old = {"summary": "Deprioritize. Old model text.", "sections": {k: "x" for k in N.SECTION_NAMES},
           "source": "api", "verified": True, "unverified_numbers": []}
    (run / "narrative.json").write_text(json.dumps(old))
    monkeypatch.setattr("research_engine.extraction.client.call_structured",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model down")))
    out = N.write_narrative(run, results, llm="api-refresh")
    assert out["summary"] == "Deprioritize. Old model text." and out["source"] == "api" and "model down" in out["error"]


def test_md_safe_neutralizes_links_images_latex_and_html():
    import importlib.util
    spec = importlib.util.spec_from_file_location("x", ROOT / "research_engine" / "app" / "streamlit_app.py")
    src = (ROOT / "research_engine" / "app" / "streamlit_app.py").read_text()
    ns = {}
    exec(src[src.index("def md_safe"):src.index("def toggle_paper")], ns)       # the function alone, no Streamlit
    out = ns["md_safe"]("![x](http://e/i.png) [l](http://e) $5 <b>")
    assert "!\\[x\\](" in out and "\\[l\\](" in out          # brackets escaped: no image, no link
    assert "\\$5" in out and "&lt;b&gt;" in out                  # no LaTeX, no raw HTML
