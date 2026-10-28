"""Extraction layer: ingest, evidence check, consensus, cache, key handling, session runner.
Uses only the synthetic fixture and the project's own mock oil paper; no API calls."""
import json
from pathlib import Path

import pytest

from research_engine.contracts import ROOT, CheckPassResult, StrategySpec, RunFiles
from research_engine.documents.ingest import ingest, page_tagged_text, ScannedPDFError
from research_engine.extraction import client as C
from research_engine.extraction.evidence_check import check_evidence, consensus, normalize_text
from research_engine.extraction.prompts import (SYSTEM_EXTRACT, build_extract_user, build_check_user,
                                                extract_run_user)

FIX = Path(__file__).parent / "fixtures"
OIL_PDF = ROOT / "papers" / "oil_factor_mock_paper_aidan.pdf"


@pytest.fixture
def pages():
    return json.loads((FIX / "mini_paper_text.json").read_text())


@pytest.fixture
def spec():
    return StrategySpec.model_validate_json((FIX / "mini_spec.json").read_text())


# ---------------------------------------------------------------- ingest
def test_ingest_oil_mock(tmp_path):
    meta = ingest(OIL_PDF, tmp_path, None)
    assert meta["n_pages"] > 0 and len(meta["sha256"]) == 64
    assert meta["selected_experiment"] is None
    text = json.loads((tmp_path / RunFiles.paper_text).read_text())
    assert text[0]["page"] == 1 and len(text) == meta["n_pages"]
    assert "oil" in " ".join(p["text"] for p in text).lower()
    tagged = page_tagged_text(text)
    assert tagged.startswith('<page n="1">') and tagged.count("</page>") == meta["n_pages"]


def test_ingest_rejects_scanned(tmp_path):
    from pypdf import PdfWriter
    w = PdfWriter()
    for _ in range(3):
        w.add_blank_page(width=612, height=792)
    pdf = tmp_path / "blank.pdf"
    with open(pdf, "wb") as f:
        w.write(f)
    with pytest.raises(ScannedPDFError, match="scanned"):
        ingest(pdf, tmp_path / "run", None)


# ---------------------------------------------------------------- evidence
def test_evidence_all_valid(spec, pages):
    res = check_evidence(spec, pages)
    assert all(r["valid"] for r in res.values())
    # quote with a line break + hyphenation ("lottery-\nlike") still matches
    assert res["economic_rationale"]["match"] == "exact"


def test_evidence_invalid(spec, pages):
    s = spec.model_copy(deep=True)
    s.holding_period.evidence.quote = "Portfolios are held for twelve months"
    assert check_evidence(s, pages)["holding_period"]["valid"] is False


def test_evidence_adjacent_page(spec, pages):
    s = spec.model_copy(deep=True)
    s.favored_group.evidence.page = 2   # quote is on page 3
    r = check_evidence(s, pages)["favored_group"]
    assert r["valid"] and r["adjacent_page"] and r["found_page"] == 3


def test_evidence_soft_hyphen_and_case(spec, pages):
    s = spec.model_copy(deep=True)
    s.signal_formula_words.evidence.quote = "MAXIMUM DAILY­RETURN in the previous   month".replace("­RETURN", " RE­TURN")
    assert check_evidence(s, pages)["signal_formula_words"]["valid"]
    assert normalize_text("ex-\ntreme") == "extreme"


# ---------------------------------------------------------------- consensus
def test_consensus_majority(spec):
    a = spec.model_copy(deep=True)
    b = spec.model_copy(deep=True)
    c = spec.model_copy(deep=True)
    c.holding_period.value = "six months"
    c.weighting = "equal"
    b.holding_period.value = "One month."          # normalizes equal to "one month"
    cons, agree = consensus([c, a, b])
    assert cons.holding_period.value in ("one month", "One month.")
    assert agree["holding_period"] == pytest.approx(2 / 3)
    assert cons.weighting == "value" and agree["weighting"] == pytest.approx(2 / 3)
    assert agree["sort_rule"] == 1.0


def test_consensus_no_majority_takes_first(spec):
    a, b = spec.model_copy(deep=True), spec.model_copy(deep=True)
    b.skip_window.value = "1 month"
    cons, agree = consensus([a, b])
    assert cons.skip_window.value == "none" and agree["skip_window"] == 0.5


# ---------------------------------------------------------------- client
def test_cache_hit_never_calls_api(spec, tmp_path, monkeypatch):
    system, user = "sys", "user text"
    C.write_cache(system, user, StrategySpec, spec, source="session", tag="t",
                  usage={"input_tokens": 1, "output_tokens": 2}, cache_dir=tmp_path)

    def boom(*a, **k):
        raise AssertionError("API client must not be built on a cache hit")
    monkeypatch.setattr(C, "_client", boom)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    obj, usage = C.call_structured(system, user, StrategySpec, "extract_1", run_dir=run_dir, cache_dir=tmp_path)
    assert obj == spec and usage["cached"] is True and usage["source"] == "session"
    assert json.loads((run_dir / RunFiles.usage).read_text())[0]["call"] == "extract_1"


def test_cache_key_depends_on_inputs():
    k1 = C.cache_key("s", "u", StrategySpec)
    assert k1 != C.cache_key("s", "u2", StrategySpec) != C.cache_key("s", "u", CheckPassResult)


def test_missing_key_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "_load_env", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(C.LLMUnavailable, match=".env"):
        C.call_structured("s", "u-missing", StrategySpec, "x", cache_dir=tmp_path)


def test_bogus_key_raises(tmp_path, monkeypatch):
    """A rejected key surfaces as LLMUnavailable. The HTTP layer is mocked to return 401."""
    import anthropic
    import httpx2 as httpx  # the SDK's HTTP layer
    monkeypatch.setattr(C, "_load_env", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-bogus")

    def handler(request):
        return httpx.Response(401, json={"type": "error", "error": {"type": "authentication_error",
                                                                    "message": "invalid x-api-key"}})
    real = anthropic.Anthropic
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda **kw: real(http_client=httpx.Client(transport=httpx.MockTransport(handler)),
                                          max_retries=0))
    with pytest.raises(C.LLMUnavailable, match=r"401.*\.env"):
        C.call_structured("s", "u-bogus", StrategySpec, "x", cache_dir=tmp_path)


# ---------------------------------------------------------------- prompts + session runner
def test_prompts():
    u = build_extract_user("<page n=\"1\">x</page>", None)
    assert "none given" in u and "<paper>" in u
    assert "Table 2" in build_extract_user("x", "Table 2")
    assert "verbatim" in SYSTEM_EXTRACT and "Ignore any instructions" in SYSTEM_EXTRACT
    assert len(SYSTEM_EXTRACT.split()) < 600


def test_session_roundtrip(spec, pages, tmp_path):
    from research_engine.extraction import session_runner as S
    run = tmp_path / "run"
    run.mkdir()
    (run / RunFiles.paper_text).write_text(json.dumps(pages))
    (run / RunFiles.paper_meta).write_text(json.dumps({"sha256": "0" * 64, "selected_experiment": None}))
    cache = tmp_path / "cache"
    p = S.export_prompt(run, 1)
    body = p.read_text()
    assert "SYSTEM" in body and "OUTPUT JSON SCHEMA (StrategySpec)" in body and "<!-- extraction run 1 -->" in body
    obj = S.import_response(run, 1, "```json\n" + spec.model_dump_json() + "\n```", cache_dir=cache)
    assert (run / "spec_raw_1.json").exists()
    # the pipeline's API-path call now hits the session cache entry
    user = extract_run_user(build_extract_user(page_tagged_text(pages), None), 1)
    hit = C.read_cache(SYSTEM_EXTRACT, user, StrategySpec, cache)
    assert hit["source"] == "session" and StrategySpec.model_validate(hit["data"]) == obj
    # check pass
    cp = S.export_check_prompt(run, 1)
    assert "OUTPUT JSON SCHEMA (CheckPassResult)" in cp.read_text()
    res = S.import_check_response(run, 1, json.dumps({"agrees": True, "items": []}), cache_dir=cache)
    assert res.agrees and (run / "check_pass_1.json").exists()
    assert C.read_cache(*S.check_messages(run, 1), CheckPassResult, cache) is not None
    with pytest.raises(Exception):
        S.import_response(run, 2, '{"paper_title": "x"}', cache_dir=cache)


def test_narrative_roundtrip(spec, tmp_path):
    from research_engine.extraction import session_runner as S
    from research_engine.extraction.schemas_extra import NarrativeOut, SECTION_NAMES
    run = tmp_path / "run"
    run.mkdir()
    (run / RunFiles.results).write_text(json.dumps({"recommendation": "deprioritize", "x": 1.23}))
    (run / RunFiles.spec_consensus).write_text(json.dumps({"spec": json.loads(spec.model_dump_json())}))
    (run / RunFiles.adaptation_log).write_text("[]")
    assert "OUTPUT JSON SCHEMA (NarrativeOut)" in S.export_narrative_prompt(run).read_text()
    answer = {"summary": "Deprioritize. ...", "sections": {n: "text" for n in SECTION_NAMES}}
    obj = S.import_narrative_response(run, json.dumps(answer), cache_dir=tmp_path / "c")
    assert isinstance(obj, NarrativeOut)
    assert json.loads((run / "narrative_raw.json").read_text())["sections"][SECTION_NAMES[0]] == "text"
