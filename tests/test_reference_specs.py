"""Preliminary reference specs validate against StrategySpec and work with the scorer."""
import shutil
from pathlib import Path

import pytest

from research_engine.contracts import ROOT, RunFiles, StrategySpec
from research_engine.evaluation.interpretation import score_interpretation

REF_DIR = ROOT / "papers" / "reference_specs"
SLUGS = ["max", "momentum", "seasonality", "lowvol", "high52", "frog",
         "decoy_profitability", "decoy_tsmom"]


def test_all_reference_files_exist_and_validate():
    for slug in SLUGS:
        path = REF_DIR / f"{slug}.preliminary.json"
        assert path.exists(), slug
        StrategySpec.model_validate_json(path.read_text())
    assert (REF_DIR / "README.md").exists()
    assert sorted(p.name for p in REF_DIR.glob("*.json")) == sorted(f"{s}.preliminary.json" for s in SLUGS)


def test_decoys_cannot_implement():
    for slug in ("decoy_profitability", "decoy_tsmom"):
        spec = StrategySpec.model_validate_json((REF_DIR / f"{slug}.preliminary.json").read_text())
        assert spec.can_implement is False and spec.cannot_implement_reasons


def test_score_demo_max_against_preliminary_reference(tmp_path):
    run = ROOT / "runs" / "demo_max"
    if not (run / RunFiles.spec_consensus).exists():
        pytest.skip("runs/demo_max not available")
    work = tmp_path / "demo_max"
    work.mkdir()
    for name in (RunFiles.spec_consensus, RunFiles.adaptation_log, "validation.json"):
        if (run / name).exists():
            shutil.copy(run / name, work / name)
    sc = score_interpretation(work, REF_DIR / "max.preliminary.json", write=False)
    assert isinstance(sc["agreement_rate"], float) and 0.0 <= sc["agreement_rate"] <= 1.0
    assert sc["preliminary"] is True
