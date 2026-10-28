"""Smoke tests for the Streamlit app (streamlit.testing, no browser, no pipeline run)."""
from pathlib import Path

import pytest

from research_engine.app import helpers as H

APP = str(Path(__file__).resolve().parents[1] / "research_engine" / "app" / "streamlit_app.py")
MOCK = Path(__file__).resolve().parents[1] / "runs" / "mock_app"

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
needs_mock = pytest.mark.skipif(not (MOCK / "progress.json").exists(), reason="runs/mock_app missing")


def _app():
    return AppTest.from_file(APP, default_timeout=60)


def test_app_loads_with_title():
    at = _app().run()
    assert not at.exception
    assert at.title[0].value == "Research Screening Agent"
    assert any(s.value == "Summary" for s in at.subheader)
    assert any("Select a paper or drop a PDF." in m.value for m in at.markdown)


@needs_mock
def test_mock_run_shows_nine_steps_and_summary():
    at = _app()
    at.session_state["selected_run"] = "mock_app"
    at.run()
    assert not at.exception
    steps = [m.value for m in at.markdown if 'class="steps"' in m.value]
    assert len(steps) == 1 and steps[0].count('class="step ') == 9
    assert any(s.value == "Summary" for s in at.subheader)
    assert any(m.value == "**investigate further**" for m in at.markdown)
    assert len(at.tabs) == 4


def test_paper_preview_and_queue():
    at = _app().run()
    at.button(key="paper_max").click().run()
    assert not at.exception
    at.button(key="use_max").click().run()
    assert at.session_state["queue"] == ["max"]


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
