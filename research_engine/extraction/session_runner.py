"""The no-API path: another agent acts as the model.

export_* writes a prompt file (system prompt, user content, output JSON schema)
that an agent can execute by hand. import_* validates the agent's JSON answer
against the same schema, writes the run file the pipeline reads, and writes
the cache entry call_structured would have written (source "session"), so a
later API-path run hits the cache instead of calling the model.

The system/user strings built here must equal the ones the pipeline sends;
both come from research_engine.extraction.prompts.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from research_engine.contracts import CheckPassResult, MODEL, RunFiles, StrategySpec
from research_engine.documents.ingest import page_tagged_text
from research_engine.extraction import prompts as P
from research_engine.extraction.client import append_usage, write_cache
from research_engine.extraction.schemas_extra import NarrativeOut

SEP = "=" * 78
EXTRACT_PROMPT = "extract_prompt_{k}.txt"
CHECK_PROMPT = "check_prompt_{k}.txt"
NARRATIVE_PROMPT = "narrative_prompt.txt"
NARRATIVE_RAW = "narrative_raw.json"


# ----------------------------------------------------------------------------
# message builders (shared with the API path)
# ----------------------------------------------------------------------------
def _read_json(path: Path):
    return json.loads(Path(path).read_text())


def extract_messages(run_dir: Path, k: int) -> tuple[str, str]:
    run_dir = Path(run_dir)
    meta = _read_json(run_dir / RunFiles.paper_meta)
    text = _read_json(run_dir / RunFiles.paper_text)
    base = P.build_extract_user(page_tagged_text(text), meta.get("selected_experiment"))
    return P.SYSTEM_EXTRACT, P.extract_run_user(base, k)


def check_messages(run_dir: Path, k: int) -> tuple[str, str]:
    run_dir = Path(run_dir)
    spec = StrategySpec.model_validate_json((run_dir / RunFiles.spec_raw.format(k=k)).read_text())
    text = _read_json(run_dir / RunFiles.paper_text)
    return P.SYSTEM_CHECK, P.build_check_user(spec, page_tagged_text(text))


def narrative_messages(run_dir: Path) -> tuple[str, str]:
    run_dir = Path(run_dir)
    results = _read_json(run_dir / RunFiles.results)
    cons = _read_json(run_dir / RunFiles.spec_consensus)
    log = _read_json(run_dir / RunFiles.adaptation_log) if (run_dir / RunFiles.adaptation_log).exists() else []
    return P.SYSTEM_NARRATIVE, P.build_narrative_user(results, cons, log)


# ----------------------------------------------------------------------------
# prompt file I/O
# ----------------------------------------------------------------------------
def _write_prompt_file(path: Path, system: str, user: str, schema: type[BaseModel]) -> Path:
    body = "\n".join([
        f"SESSION PROMPT for {schema.__name__} (model stand-in for {MODEL}).",
        "Act as the model: follow SYSTEM, answer USER, and return ONLY one JSON object",
        "that validates against OUTPUT JSON SCHEMA. Then import it with session_runner.",
        SEP, "SYSTEM", SEP, system,
        SEP, "USER", SEP, user,
        SEP, f"OUTPUT JSON SCHEMA ({schema.__name__})", SEP,
        json.dumps(schema.model_json_schema(), indent=1),
    ])
    path.write_text(body + "\n")
    return path


def parse_json_text(json_text: str) -> dict:
    """Accept a bare JSON object, optionally wrapped in a ```json fence."""
    s = json_text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.S)
    if m:
        s = m.group(1)
    return json.loads(s)


def _est_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _record(run_dir: Path, system: str, user: str, schema: type[BaseModel], obj: BaseModel,
            tag: str, source: str, cache_dir: Optional[Path]) -> None:
    out = obj.model_dump_json()
    usage = {"input_tokens": _est_tokens(system + user), "output_tokens": _est_tokens(out), "estimated": True}
    write_cache(system, user, schema, obj, source=source, tag=tag, usage=usage, cache_dir=cache_dir)
    append_usage(run_dir, {"call": tag, "model": MODEL, **usage, "cached": False, "source": source})


# ----------------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------------
def export_prompt(run_dir, k: int = 1) -> Path:
    run_dir = Path(run_dir)
    system, user = extract_messages(run_dir, k)
    return _write_prompt_file(run_dir / EXTRACT_PROMPT.format(k=k), system, user, StrategySpec)


def import_response(run_dir, k: int, json_text: str, source: str = "session",
                    cache_dir: Optional[Path] = None) -> StrategySpec:
    run_dir = Path(run_dir)
    obj = StrategySpec.model_validate(parse_json_text(json_text))
    system, user = extract_messages(run_dir, k)
    (run_dir / RunFiles.spec_raw.format(k=k)).write_text(obj.model_dump_json(indent=1))
    _record(run_dir, system, user, StrategySpec, obj, f"extract_{k}", source, cache_dir)
    return obj


# ----------------------------------------------------------------------------
# check pass
# ----------------------------------------------------------------------------
def export_check_prompt(run_dir, k: int = 1) -> Path:
    run_dir = Path(run_dir)
    system, user = check_messages(run_dir, k)
    return _write_prompt_file(run_dir / CHECK_PROMPT.format(k=k), system, user, CheckPassResult)


def import_check_response(run_dir, k: int, json_text: str, source: str = "session",
                          cache_dir: Optional[Path] = None) -> CheckPassResult:
    run_dir = Path(run_dir)
    obj = CheckPassResult.model_validate(parse_json_text(json_text))
    if obj.items and obj.agrees:
        raise ValueError("check pass says agrees=true but lists contradictions")
    system, user = check_messages(run_dir, k)
    (run_dir / RunFiles.check_pass.format(k=k)).write_text(obj.model_dump_json(indent=1))
    _record(run_dir, system, user, CheckPassResult, obj, f"check_{k}", source, cache_dir)
    return obj


# ----------------------------------------------------------------------------
# narrative
# ----------------------------------------------------------------------------
def export_narrative_prompt(run_dir) -> Path:
    run_dir = Path(run_dir)
    system, user = narrative_messages(run_dir)
    return _write_prompt_file(run_dir / NARRATIVE_PROMPT, system, user, NarrativeOut)


def import_narrative_response(run_dir, json_text: str, source: str = "session",
                              cache_dir: Optional[Path] = None) -> NarrativeOut:
    """Writes narrative_raw.json ({"summary", "sections"}); number verification and
    narrative.json belong to reporting/narrative.py, which will hit this cache entry."""
    run_dir = Path(run_dir)
    obj = NarrativeOut.model_validate(parse_json_text(json_text))
    system, user = narrative_messages(run_dir)
    (run_dir / NARRATIVE_RAW).write_text(json.dumps(obj.to_contract_dict(), indent=1, ensure_ascii=False))
    _record(run_dir, system, user, NarrativeOut, obj, "narrative", source, cache_dir)
    return obj
