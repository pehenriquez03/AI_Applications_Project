"""Prompt loading and user-message builders.

System prompts live in prompts/<name>_<PROMPT_VERSION>.md next to this file.
The paper text is always passed as page-tagged data inside the user turn.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from research_engine.contracts import PROMPT_VERSION

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str, version: str = PROMPT_VERSION) -> str:
    return (PROMPT_DIR / f"{name}_{version}.md").read_text().strip()


def extract_system() -> str:
    return load_prompt("extract")


def check_system() -> str:
    return load_prompt("check")


def narrative_system() -> str:
    return load_prompt("narrative")


SYSTEM_EXTRACT = extract_system()
SYSTEM_CHECK = check_system()
SYSTEM_NARRATIVE = narrative_system()


def extract_run_user(base_user: str, k: int) -> str:
    """Per-run user content: a marker the model ignores keeps the k cache entries distinct.
    Must match the string the pipeline sends (pipeline.stage_extract)."""
    return base_user + f"\n<!-- extraction run {k} -->"


def spec_json_for_check(spec) -> str:
    if isinstance(spec, str):
        return spec
    return spec.model_dump_json(indent=1)


def build_extract_user(page_tagged_text: str, selected_experiment: Optional[str]) -> str:
    if selected_experiment:
        sel = f"Selected experiment: {selected_experiment}"
    else:
        sel = ("Selected experiment: none given. Identify the paper's main experiment and name it "
               "in selected_experiment.")
    return (f"{sel}\n\nExtract the strategy from the paper below.\n\n"
            f"<paper>\n{page_tagged_text}\n</paper>")


def build_check_user(spec, page_tagged_text: str) -> str:
    """spec: a StrategySpec or its JSON string."""
    return (f"<strategy_spec>\n{spec_json_for_check(spec)}\n</strategy_spec>\n\n"
            f"Check the specification above against the paper below.\n\n"
            f"<paper>\n{page_tagged_text}\n</paper>")


def build_narrative_user(results: dict, consensus_spec: dict, adaptation_log: list) -> str:
    return ("<results_json>\n" + json.dumps(results, indent=1, ensure_ascii=False, default=str)
            + "\n</results_json>\n\n<consensus_spec>\n"
            + json.dumps(consensus_spec, indent=1, ensure_ascii=False, default=str)
            + "\n</consensus_spec>\n\n<adaptation_log>\n"
            + json.dumps(adaptation_log, indent=1, ensure_ascii=False, default=str)
            + "\n</adaptation_log>\n\nWrite the narrative.")
