"""Smoke tests for the Streamlit app (streamlit.testing, no browser, no pipeline run)."""
from pathlib import Path

import pytest

from research_engine.app import helpers as H

APP = str(Path(__file__).resolve().parents[1] / "research_engine" / "app" / "streamlit_app.py")
RUNS = Path(__file__).resolve().parents[1] / "runs"
MOCK = RUNS / "mock_app"

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
needs_mock = pytest.mark.skipif(not (MOCK / "progress.json").exists(), reason="runs/mock_app missing")


def _app():
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["key_skipped"] = True          # the key setup gate is tested on its own
    return at


def test_key_gate_shows_first_and_can_be_skipped():
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["force_gate"] = True
    at.run()
    assert not at.exception
    gate = [m.value for m in at.markdown if 'class="gate"' in m.value]
    assert len(gate) == 1 and "shared in class" in gate[0]
    assert at.radio(key="key_which").value == "Class key (default)"
    assert at.button(key="key_save").disabled                      # nothing pasted yet
    at.button(key="key_skip").click().run()
    assert not at.exception
    assert not [m.value for m in at.markdown if 'class="gate"' in m.value]
    assert any("Select a paper or drop a PDF." in m.value for m in at.markdown)


def test_app_loads_with_title():
    at = _app().run()
    assert not at.exception
    top = [m.value for m in at.markdown if 'class="topbar"' in m.value]
    assert len(top) == 1 and "WARREN" in top[0] and "RESEARCH PAPER SCREENING AGENT" in top[0]
    assert any('class="chips"' in m.value and "2015-01-01 → 2025-12-31" in m.value for m in at.markdown)
    assert any("Summary" in m.value and 'class="ph"' in m.value for m in at.markdown)
    assert any("Select a paper or drop a PDF." in m.value for m in at.markdown)
    assert len([b for b in at.button if b.key.startswith("paper_")]) == 9      # one tile per registry paper
    assert at.selectbox(key="model").value == "claude-sonnet-5-5"
    assert tuple(at.select_slider(key="window").value) == ("2015-01", "2025-12")


@needs_mock
def test_mock_run_shows_nine_steps_and_summary():
    at = _app()
    at.session_state["selected_run"] = "mock_app"
    at.run()
    assert not at.exception
    steps = [m.value for m in at.markdown if 'class="steps"' in m.value]
    assert len(steps) == 1 and steps[0].count('class="step ') == 9
    assert any("Summary" in m.value and 'class="ph"' in m.value for m in at.markdown)
    assert any('class="verdict' in m.value and "Investigate Further" in m.value for m in at.markdown)
    assert len(at.tabs) == 5


def _open_run(name: str):
    at = _app()
    at.session_state["selected_run"] = name
    at.run()
    assert not at.exception
    return at


@pytest.mark.skipif(not (RUNS / "demo_max" / "progress.json").exists(), reason="runs/demo_max missing")
def test_real_run_demo_max_renders_all_panels():
    at = _open_run("demo_max")
    steps = [m.value for m in at.markdown if 'class="steps"' in m.value]
    assert len(steps) == 1 and steps[0].count('class="step ') == 9
    assert any("Summary" in m.value and 'class="ph"' in m.value for m in at.markdown)
    rec = H.read_progress(RUNS / "demo_max")["recommendation"]
    assert any('class="verdict' in m.value and rec.title() in m.value for m in at.markdown)
    order = [m.value for m in at.markdown if 'class="ph"' in m.value]
    assert "Summary" in order[0] and any("Pipeline" in v for v in order[1:])   # summary first, steps later
    assert len(at.tabs) == 5
    assert len(at.table[0].value) >= 10                      # extraction table: one row per material field
    heads = [m.value for m in at.markdown if m.value.startswith("**")]
    for h in ("**Performance**", "**Trading cost grid**", "**Sub-periods**", "**Ranked groups**", "**Share of paths**"):
        assert h in heads
    assert any("Spearman rho" in c.value for c in at.caption)


@pytest.mark.skipif(not (RUNS / "demo_decoy_tsmom" / "progress.json").exists(), reason="runs/demo_decoy_tsmom missing")
def test_blocked_decoy_says_unable_to_evaluate():
    at = _open_run("demo_decoy_tsmom")
    assert any('class="verdict' in m.value and "Unable to Evaluate" in m.value for m in at.markdown)
    assert len(at.tabs) == 5                                   # tabs tolerate the missing evaluation/economics files


def test_paper_label_prefers_registry():
    known = {"sha": {"abc": "Bali, Cakici & Whitelaw 2011 — MAX"}, "file": {"x.pdf": "Novy-Marx 2013 — profitability (decoy)"}}
    assert H.paper_label({"sha256": "abc", "title_guess": "T"}, known) == "Bali 2011 — MAX"
    assert H.paper_label({"sha256": "zzz", "file_name": "x.pdf"}, known) == "Novy-Marx 2013 — profitability"
    assert H.paper_label({"title_guess": "SOME TITLE:"}, known) == "SOME TITLE"


def test_list_runs_includes_demo_and_dev(tmp_path):
    for n in ("demo_a", "dev_b", "mock_c", "_d"):
        (tmp_path / n).mkdir()
        (tmp_path / n / "progress.json").write_text('{"stages": [], "recommendation": "deprioritize", "paper_title": "T"}')
    assert sorted(r["id"] for r in H.list_runs(tmp_path)) == ["demo_a", "dev_b"]


def test_paper_tile_toggles_queue_and_previews():
    at = _app().run()
    at.button(key="paper_max").click().run()
    assert not at.exception
    assert at.session_state["queue"] == ["max"]
    assert at.session_state["preview"] == "max"
    assert "Bali" in at.button(key="paper_max").help                     # hover preview is the button tooltip
    at.button(key="paper_momentum").click().run()                        # one paper at a time: replaces the selection
    assert at.session_state["queue"] == ["momentum"]
    at.button(key="paper_momentum").click().run()                        # second click deselects
    assert at.session_state["queue"] == []


def test_run_header_shows_window_chip():
    at = _open_run("demo_momentum") if (RUNS / "demo_momentum" / "progress.json").exists() else None
    if at is None:
        pytest.skip("runs/demo_momentum missing")
    heads = [m.value for m in at.markdown if 'class="runh"' in m.value]
    assert len(heads) == 1 and "BACKTEST 2015-01-01 → 2025-12-31" in heads[0]
    assert any('class="quote"' in m.value and "Net CAGR" in m.value for m in at.markdown)


def test_helpers_tolerate_missing_files(tmp_path):
    assert H.read_json(tmp_path / "nope.json") is None
    assert H.read_progress(tmp_path) is None
    assert H.list_runs(tmp_path) == []
    (tmp_path / "_uploads").mkdir()
    (tmp_path / "mock_x").mkdir()
    (tmp_path / "mock_x" / "progress.json").write_text("{}")
    assert H.list_runs(tmp_path) == []


def test_table_sections_nested():
    obj = {"a": 1.23456, "grid": {"0": {"x": 1, "y": 2.5}, "10": {"x": 3, "y": None}}, "deep": {"k": {"m": {"z": 1}}}}
    secs = H.table_sections(obj)
    assert len(secs) >= 3
    assert all(df.map(lambda v: isinstance(v, str)).all().all() for _, df in secs)
