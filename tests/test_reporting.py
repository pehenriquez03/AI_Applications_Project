"""Reporting stage: results.json, rubric, narrative verification, HTML rendering.

Runs on a synthetic run folder built by research_engine.reporting.mock (the
same generator that writes runs/mock_report). No model calls.
"""
from __future__ import annotations

import copy
import json
import shutil

import pytest

from research_engine.contracts import RECOMMENDATIONS, RunFiles
from research_engine.reporting import narrative as N
from research_engine.reporting.mock import build_mock_blocked, build_mock_run
from research_engine.reporting.render import render_blocked, render_dossier, render_index
from research_engine.reporting.results import assemble_results
from research_engine.reporting.rubric import RULES, recommend

TOP_KEYS = {"paper", "extraction", "adaptation", "data", "construction", "performance", "robustness",
            "economics", "interpretation", "limitations", "llm_usage"}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    build_mock_run(root / "mock_report")
    return root


@pytest.fixture()
def run_dir(runs, tmp_path):
    d = tmp_path / "runs" / "mock_report"
    shutil.copytree(runs / "mock_report", d)
    return d


@pytest.fixture(scope="module")
def base_results(runs):
    r = assemble_results(runs / "mock_report")
    rec = recommend(r)
    r.update(rubric=rec["rubric"], recommendation=rec["recommendation"], recommendation_reasons=rec["reasons"])
    return r


# ----------------------------------------------------------------------------
# results.json
# ----------------------------------------------------------------------------
def test_assemble_results_keys(base_results):
    r = base_results
    assert TOP_KEYS <= set(r)
    assert r["status"] == "complete"
    assert r["paper"]["title"].startswith("[MOCK]")
    assert r["adaptation"]["n_assumed"] == 2 and r["adaptation"]["n_adapted"] == 3
    assert r["extraction"]["evidence_valid_count"] == 15
    assert r["extraction"]["n_check_disagreements"] == 1
    assert r["construction"]["n_skipped_months"] == 1 and len(r["construction"]["sample_top10"]) == 10
    h = r["performance"]["headline"]
    assert h["strategy_net_cagr"] is not None and h["alpha_t"] is not None
    assert set(r["robustness"]["holds_at_bps"]) == {"0", "10", "25"}
    assert r["economics"]["headline"]["incremental_fees_pv_quantiles"]["p50"] is not None
    assert r["llm_usage"]["n_calls"] == 6
    assert r["llm_usage"]["cost_usd"] == pytest.approx(375000 * 2 / 1e6 + 12300 * 10 / 1e6, rel=1e-4)
    assert any("2 fields assumed" in x for x in r["limitations"])
    assert any("Survivor universe" in x and "19 excluded" in x for x in r["limitations"])
    json.dumps(r)  # JSON-safe


def test_assemble_results_missing_files(tmp_path):
    d = tmp_path / "bare"
    d.mkdir()
    (d / RunFiles.paper_meta).write_text(json.dumps({"title": "Only meta", "sha256": "0" * 64}))
    r = assemble_results(d)
    assert r["paper"]["title"] == "Only meta"
    for k in ("extraction", "adaptation", "construction", "performance", "robustness", "economics",
              "interpretation", "llm_usage"):
        assert r[k] is None
    assert r["status"] == "blocked"
    assert recommend(r)["recommendation"] == "unable to evaluate"


# ----------------------------------------------------------------------------
# rubric
# ----------------------------------------------------------------------------
def _scenario(base, *, perf=None, holds=None, subs=None, rho=True, frac=None, n_assumed=None, status=None):
    r = copy.deepcopy(base)
    for k in ("rubric", "recommendation", "recommendation_reasons"):
        r.pop(k, None)
    h = r["performance"]["headline"]
    if perf == "positive":
        h.update(strategy_net_cagr=0.12, ew_universe_cagr=0.09, strategy_net_sharpe=0.8, ew_universe_sharpe=0.6)
    elif perf == "negative":
        h.update(strategy_net_cagr=0.05, ew_universe_cagr=0.09, strategy_net_sharpe=0.3, ew_universe_sharpe=0.6)
    if holds is not None:
        r["robustness"]["holds_at_bps"] = {"0": holds, "10": holds, "25": holds}
    if subs is not None:
        r["robustness"]["holds_in_subperiod"] = {"2015-2019": subs[0], "2020-2025": subs[1]}
    r["robustness"]["rho_in_expected_direction"] = rho
    if frac is not None:
        r["economics"]["frac_npv_positive"] = frac
    if n_assumed is not None:
        r["adaptation"]["n_assumed"] = n_assumed
    if status is not None:
        r["status"] = status
    return r


def test_recommend_pursue(base_results):
    r = _scenario(base_results, perf="positive", holds=True, subs=(True, True), frac=0.72, n_assumed=1)
    out = recommend(r)
    assert out["rubric"]["flags"] == {"implementation_reliability": "green", "historical_performance": "positive",
                                      "robustness": "strong", "economics": "positive", "limitations_count": 4}
    assert out["recommendation"] == "pursue"
    assert 3 <= len(out["reasons"]) <= 4 and "12.00%" in out["reasons"][0]


def test_recommend_investigate_further(base_results):
    # strong evidence everywhere except reliability (7 assumed -> red)
    r = _scenario(base_results, perf="positive", holds=True, subs=(True, True), frac=0.72, n_assumed=7)
    out = recommend(r)
    assert out["rubric"]["flags"]["implementation_reliability"] == "red"
    assert out["recommendation"] == "investigate further"
    # the mock itself: moderate robustness
    assert recommend(_scenario(base_results))["recommendation"] == "investigate further"


def test_recommend_deprioritize(base_results):
    assert recommend(_scenario(base_results, perf="negative", frac=0.9))["recommendation"] == "deprioritize"
    r = _scenario(base_results, perf="positive", holds=False, subs=(False, False), frac=0.2)
    out = recommend(r)
    assert out["rubric"]["flags"]["robustness"] == "weak" and out["rubric"]["flags"]["economics"] == "negative"
    assert out["recommendation"] == "deprioritize"


def test_recommend_unable(base_results):
    r = _scenario(base_results, status="blocked")
    r["status_reasons"] = ["required input missing: market cap"]
    out = recommend(r)
    assert out["recommendation"] == "unable to evaluate"
    assert "market cap" in out["reasons"][0]


def test_rubric_declares_rules(base_results):
    out = recommend(base_results)
    assert out["rubric"]["rules"] == RULES
    assert out["recommendation"] in RECOMMENDATIONS
    assert "reported, not a gate" in RULES


# ----------------------------------------------------------------------------
# narrative
# ----------------------------------------------------------------------------
def test_template_narrative_verifies(base_results):
    nar = N.template_narrative(base_results)
    ok, bad = N.verify_numbers(nar, base_results)
    assert ok, bad
    assert set(nar["sections"]) == set(N.SECTION_NAMES)


def test_bogus_number_flagged(base_results):
    nar = N.template_narrative(base_results)
    nar["sections"][N.SECTION_NAMES[2]] += " The strategy earned 47.3% a year and $987.6M in fees."
    ok, bad = N.verify_numbers(nar, base_results)
    assert not ok
    assert "47.3%" in bad and "$987.6M" in bad


def test_number_forms_match(base_results):
    cagr = base_results["performance"]["headline"]["strategy_net_cagr"]
    p = base_results["economics"]["frac_npv_positive"]
    q = base_results["economics"]["quantiles"]["incremental_fees_pv"]["p50"]
    text = f"CAGR {cagr * 100:.2f}% and P(NPV > 0) of {p * 100:.0f} percent; median fees ${q / 1e6:.2f} million; 2015-2019."
    ok, bad = N.verify_numbers({"summary": text, "sections": {}}, base_results)
    assert ok, bad


def test_write_narrative_template_and_prompt(run_dir, base_results):
    out = N.write_narrative(run_dir, base_results, llm="files_only")
    assert out["source"] == "template" and out["verified"] is True and out["unverified_numbers"] == []
    assert out["label"] == N.TEMPLATE_LABEL
    saved = json.loads((run_dir / RunFiles.narrative).read_text())
    assert saved["source"] == "template"
    prompt = (run_dir / N.PROMPT_FILE).read_text()
    assert "=== SYSTEM ===" in prompt and "<results_json>" in prompt and "What the backtest found" in prompt


def test_write_narrative_prefers_session_file(run_dir, base_results):
    sections = {s: "Text without figures." for s in N.SECTION_NAMES}
    sections[N.SECTION_NAMES[3]] = "P(NPV > 0) was 99.9%."
    (run_dir / RunFiles.narrative).write_text(json.dumps({"summary": "Investigate further. Fine.", "sections": sections}))
    out = N.write_narrative(run_dir, base_results, llm="files_only")
    assert out["source"] == "session" and out["label"] == "model-written"
    assert out["verified"] is False and out["unverified_numbers"] == ["99.9%"]


def test_write_narrative_api_unavailable_falls_back(run_dir, base_results, monkeypatch):
    from research_engine.extraction import client

    def boom(*a, **k):
        raise client.LLMUnavailable("no key")
    monkeypatch.setattr(client, "call_structured", boom)
    out = N.write_narrative(run_dir, base_results, llm="api")
    assert out["source"] == "template" and "LLMUnavailable" in out["error"]


# ----------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------
def test_dossier_renders(run_dir, base_results):
    nar = N.write_narrative(run_dir, base_results, llm="files_only")
    p = render_dossier(run_dir, base_results, nar)
    html = p.read_text()
    assert p.name == "dossier.html"
    assert base_results["recommendation"] in html
    assert "{{" not in html and "{%" not in html
    assert html.count("data:image/png;base64,") >= 6
    assert 'src="http' not in html and "<link" not in html
    assert "template narrative; model narrative pending" in html
    assert 'href="returns.parquet"' in html and "preliminary" in html
    order = ["id=\"recommendation\"", "id=\"summary\"", "id=\"paper\"", "id=\"implemented\"", "id=\"data\"",
             "id=\"construction\"", "id=\"performance\"", "id=\"robustness\"", "id=\"economics\"",
             "id=\"interpretation\"", "id=\"narrative\"", "id=\"downloads\""]
    pos = [html.index(s) for s in order]
    assert pos == sorted(pos)


def test_dossier_renders_with_missing_sections(tmp_path, base_results):
    d = tmp_path / "partial"
    d.mkdir()
    r = copy.deepcopy(base_results)
    for k in ("performance", "robustness", "economics", "construction", "interpretation"):
        r[k] = None
    html = render_dossier(d, r, None).read_text()
    assert "No performance results on file." in html and "{{" not in html


def test_index_renders(run_dir, base_results):
    (run_dir / RunFiles.results).write_text(json.dumps(base_results))
    nar = N.write_narrative(run_dir, base_results, llm="files_only")
    render_dossier(run_dir, base_results, nar)
    p = render_index(run_dir.parent)
    html = p.read_text()
    assert p.name == "index.html"
    assert "mock_report/dossier.html" in html and "[MOCK]" in html and "{{" not in html


def test_render_blocked(tmp_path):
    d = build_mock_blocked(tmp_path / "decoy")
    summary = render_blocked(d, "can_implement is false: requires accounting data")
    assert summary.startswith("Unable to evaluate.")
    html = (d / RunFiles.dossier).read_text()
    assert "unable to evaluate" in html.lower() and "requires accounting data" in html
    assert "What the paper proposes" in html and "{{" not in html
    res = json.loads((d / RunFiles.results).read_text())
    assert res["recommendation"] == "unable to evaluate" and res["status"] == "blocked"
