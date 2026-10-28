"""The only path to the model: structured calls with a disk cache.

`call_structured` returns a validated Pydantic object and a usage dict. Every
answer is cached under CACHE_DIR keyed by model, prompt version, schema
version, system prompt, user content and schema name. A cache hit never calls
the API. The session runner (session_runner.py) writes the same cache entries
when another agent stands in for the model, so callers cannot tell the two
apart except by the entry's "source" field.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from research_engine.contracts import (
    CACHE_DIR, MODEL, PROMPT_VERSION, ROOT, RunFiles, SCHEMA_VERSION,
)

ENV_FILE = ROOT / ".env"


class LLMUnavailable(RuntimeError):
    """The model cannot be called (key missing or rejected)."""


class LLMRefusal(RuntimeError):
    """The model refused to answer."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cache_key(system: str, user: str, schema: type[BaseModel], model: Optional[str] = None) -> str:
    raw = "|".join([model or MODEL, PROMPT_VERSION, SCHEMA_VERSION, system, user, schema.__name__])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_path(key: str, cache_dir: Optional[Path] = None) -> Path:
    return Path(cache_dir or CACHE_DIR) / f"{key}.json"


def read_cache(system: str, user: str, schema: type[BaseModel],
               cache_dir: Optional[Path] = None, model: Optional[str] = None) -> Optional[dict]:
    p = cache_path(cache_key(system, user, schema, model), cache_dir)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def write_cache(system: str, user: str, schema: type[BaseModel], obj: BaseModel, *,
                source: str, tag: str, usage: dict, cache_dir: Optional[Path] = None,
                model: Optional[str] = None) -> Path:
    p = cache_path(cache_key(system, user, schema, model), cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = {"source": source, "model": model or MODEL, "tag": tag, "created": _now(),
             "usage": usage, "data": json.loads(obj.model_dump_json())}
    p.write_text(json.dumps(entry, indent=1, ensure_ascii=False))
    return p


def append_usage(run_dir: Optional[Path], record: dict) -> None:
    if run_dir is None:
        return
    p = Path(run_dir) / RunFiles.usage
    rows = json.loads(p.read_text()) if p.exists() else []
    rows.append(record)
    p.write_text(json.dumps(rows, indent=1))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE, override=False)
    except ImportError:  # pragma: no cover
        pass


def _client():
    """Build the Anthropic client; raise LLMUnavailable if no key is set."""
    _load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise LLMUnavailable(f"ANTHROPIC_API_KEY is not set; add it to {ENV_FILE}")
    import anthropic
    return anthropic.Anthropic()


def call_structured(system: str, user: str, schema: type[BaseModel], tag: str,
                    run_dir: Optional[Path] = None, *, cache_dir: Optional[Path] = None,
                    max_tokens: int = 16000, model: Optional[str] = None,
                    refresh: bool = False) -> tuple[BaseModel, dict]:
    """One structured model call, cached. Returns (validated object, usage dict).
    model: Claude model id; defaults to contracts.MODEL. The cache is keyed by model.
    refresh: ignore an existing cache entry (e.g. one a session agent wrote) and call the API; the
    new answer overwrites the entry."""
    model = model or MODEL
    hit = None if refresh else read_cache(system, user, schema, cache_dir, model)
    if hit is not None:
        obj = schema.model_validate(hit["data"])
        u = hit.get("usage") or {}
        usage = {"call": tag, "model": hit.get("model", model),
                 "input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0),
                 "cached": True, "source": hit.get("source", "api")}
        append_usage(run_dir, usage)
        return obj, usage

    import anthropic
    client = _client()
    try:
        resp = client.messages.parse(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}], output_format=schema,
        )
    except anthropic.AuthenticationError as e:
        raise LLMUnavailable(
            f"Anthropic rejected ANTHROPIC_API_KEY (401). Replace the key in {ENV_FILE}. Detail: {e}"
        ) from e
    except anthropic.PermissionDeniedError as e:
        raise LLMUnavailable(
            f"Anthropic denied access for this key (403). Check the key in {ENV_FILE}. Detail: {e}"
        ) from e

    if resp.stop_reason == "refusal":
        raise LLMRefusal(f"model refused the {tag} call")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError(f"{tag}: output truncated at max_tokens={max_tokens}")
    obj = resp.parsed_output
    if obj is None:
        raise RuntimeError(f"{tag}: response did not parse into {schema.__name__}")

    usage = {"call": tag, "model": model,
             "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
             "cached": False, "source": "api"}
    write_cache(system, user, schema, obj, source="api", tag=tag,
                usage={"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]},
                cache_dir=cache_dir, model=model)
    append_usage(run_dir, usage)
    return obj, usage
