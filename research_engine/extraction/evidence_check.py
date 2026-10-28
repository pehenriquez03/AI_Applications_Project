"""Evidence verification and run-to-run consensus.

check_evidence: every evidenced field's quote must appear on its cited page
(after normalizing whitespace, soft hyphens, line-break hyphenation, ligatures,
quote characters and case). A quote found only on the page before or after is
still valid but flagged "adjacent_page". A second, looser match that also
ignores all spaces and hyphens tolerates PDF extraction artifacts and is
flagged match="loose".

consensus: per material field, the majority value across the k extractions
(string equality after normalization), else the first run's value.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

from research_engine.contracts import MATERIAL_FIELDS, Evidence, StrategySpec

# fields whose evidence lives in a sibling "<field>_evidence" attribute
_SIBLING_EVIDENCE = {"sort_rule": "sort_rule_evidence", "weighting": "weighting_evidence",
                     "expected_sign": "expected_sign_evidence"}
# non-material fields that also take a majority vote
EXTRA_CONSENSUS_FIELDS = ["can_implement", "selected_experiment"]

_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'",
                            "“": '"', "”": '"', "„": '"', "′": "'",
                            "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-"})


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("­", "")                              # soft hyphen
    s = s.translate(_QUOTE_MAP)
    s = re.sub(r"(\w)-[ \t]*\n[ \t]*(\w)", r"\1\2", s)        # line-break hyphenation
    s = re.sub(r"\s+", " ", s)
    return s.strip().casefold()


def _loose(s: str) -> str:
    return re.sub(r"[\s\-]+", "", normalize_text(s))


def _evidence_of(spec: StrategySpec, field: str) -> Evidence:
    if field in _SIBLING_EVIDENCE:
        return getattr(spec, _SIBLING_EVIDENCE[field])
    return getattr(spec, field).evidence


def _find(quote: str, pages: dict[int, str], page: int):
    """Return (match, found_page) with match in {"exact", "loose", None}."""
    q, ql = normalize_text(quote), _loose(quote)
    if not q:
        return None, None
    order = [page, page - 1, page + 1]
    for kind in ("exact", "loose"):
        for pg in order:
            if pg not in pages:
                continue
            hay = normalize_text(pages[pg]) if kind == "exact" else _loose(pages[pg])
            needle = q if kind == "exact" else ql
            if needle and needle in hay:
                return kind, pg
    return None, None


def check_evidence(spec: StrategySpec, paper_text: list[dict]) -> dict:
    """field -> {"valid", "page", "found_page", "adjacent_page", "match"}."""
    pages = {int(p["page"]): p["text"] for p in paper_text}
    out = {}
    for field in MATERIAL_FIELDS:
        ev = _evidence_of(spec, field)
        match, found = _find(ev.quote, pages, ev.page)
        out[field] = {"valid": match is not None, "page": ev.page, "found_page": found,
                      "adjacent_page": bool(found is not None and found != ev.page),
                      "match": match}
    return out


def evidence_summary(result: dict) -> dict:
    n = len(result)
    valid = sum(r["valid"] for r in result.values())
    return {"n_fields": n, "n_valid": valid, "frac_valid": valid / n if n else 0.0,
            "n_adjacent": sum(r["adjacent_page"] for r in result.values()),
            "invalid_fields": [f for f, r in result.items() if not r["valid"]]}


# ----------------------------------------------------------------------------
# consensus
# ----------------------------------------------------------------------------
def _norm_value(v) -> str:
    if isinstance(v, list):
        return " | ".join(sorted(normalize_text(str(x)).rstrip(".;, ") for x in v))
    if isinstance(v, bool):
        return str(v)
    return normalize_text(str(v)).rstrip(".;, ")


def _field_value(spec: StrategySpec, field: str):
    v = getattr(spec, field)
    return v.value if hasattr(v, "value") else v


def consensus(specs: list[StrategySpec]) -> tuple[StrategySpec, dict]:
    """(consensus spec, agreement: field -> fraction of runs matching the chosen value)."""
    if not specs:
        raise ValueError("consensus needs at least one spec")
    n = len(specs)
    data = specs[0].model_dump()
    agreement: dict[str, float] = {}
    for field in MATERIAL_FIELDS + EXTRA_CONSENSUS_FIELDS:
        keys = [_norm_value(_field_value(s, field)) for s in specs]
        top, count = Counter(keys).most_common(1)[0]
        chosen_key = top if count > n / 2 else keys[0]
        src = specs[keys.index(chosen_key)]
        src_dump = src.model_dump()
        data[field] = src_dump[field]
        if field in _SIBLING_EVIDENCE:
            data[_SIBLING_EVIDENCE[field]] = src_dump[_SIBLING_EVIDENCE[field]]
        if field == "can_implement":
            data["cannot_implement_reasons"] = src_dump["cannot_implement_reasons"]
        agreement[field] = sum(k == chosen_key for k in keys) / n
    # unresolved details: union in first-seen order
    seen, merged = set(), []
    for s in specs:
        for d in s.unresolved_details:
            key = normalize_text(d)
            if key not in seen:
                seen.add(key)
                merged.append(d)
    data["unresolved_details"] = merged
    return StrategySpec.model_validate(data), agreement
