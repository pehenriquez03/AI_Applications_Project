"""Interpretation scorecard: how reliable was the extraction?

`score_interpretation(run_dir, reference_spec_path)` reads spec_consensus.json,
validation.json and adaptation_log.json from the run folder and, when a
reference StrategySpec JSON path is supplied, compares the material fields to it.
This module never chooses or locates a reference itself; the caller decides.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from research_engine.contracts import MATERIAL_FIELDS, RunFiles

VALIDATION_FILE = "validation.json"   # not in RunFiles yet
CRITICAL_FIELDS = ("signal_formula_words", "formation_window", "holding_period",
                   "sort_rule", "favored_group", "expected_sign")


def _load(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def _norm_str(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z.%+\-]+", " ", str(s).lower())).strip(" .")


def _normalize(v):
    """Comparable form of a StrategySpec field: str, or sorted tuple of str for lists."""
    if isinstance(v, dict) and "value" in v:
        v = v["value"]
    if isinstance(v, (list, tuple)):
        return tuple(sorted(_norm_str(x) for x in v))
    return _norm_str(v)


_ENUM_FIELDS = {"sort_rule", "weighting", "expected_sign"}
_STOP = {"the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "with", "by", "at", "is", "are",
         "its", "each", "per", "from", "over", "stock", "stocks", "return", "returns", "portfolio", "portfolios"}


def _tokens(x) -> set:
    if isinstance(x, (list, tuple)):
        x = " ".join(x)
    return {t for t in re.split(r"[^0-9a-z%]+", str(x)) if t and t not in _STOP}


def _months(x) -> set:
    words = {"one": 1, "two": 2, "three": 3, "six": 6, "twelve": 12}
    out = set()
    for m in re.finditer(r"(\d+|one|two|three|six|twelve)[- ]?(month|week|year|day)", str(x)):
        n = int(m.group(1)) if m.group(1).isdigit() else words[m.group(1)]
        unit = m.group(2)
        out.add(n * (12 if unit == "year" else 1) if unit in ("month", "year") else (n, unit))
    return out


def _values_match(ours, ref, field: str) -> bool:
    """Exact for enums; for free text, substring containment or token Jaccard >= 0.5,
    and for window-like fields agreement on the stated number of months."""
    if field in _ENUM_FIELDS or isinstance(ours, tuple) and isinstance(ref, tuple) and ours == ref:
        return ours == ref
    a, b = (" ".join(ours) if isinstance(ours, tuple) else ours), (" ".join(ref) if isinstance(ref, tuple) else ref)
    if a == b or (a and b and (a in b or b in a)):
        return True
    ta, tb = _tokens(a), _tokens(b)
    if field in ("formation_window", "skip_window", "holding_period", "rebalance_frequency"):
        ma, mb = _months(a), _months(b)
        if ma or mb:
            return ma == mb
        if {"none", "no", "0"} & ta and {"none", "no", "0"} & tb:
            return True
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= 0.5


def _unwrap_spec(d: dict) -> dict:
    return d["spec"] if isinstance(d, dict) and "spec" in d and "paper_title" not in d else d


def compare_to_reference(spec: dict, reference: dict) -> dict:
    fields, crit, minor, n_cmp, n_match = {}, [], [], 0, 0
    for f in MATERIAL_FIELDS:
        if f not in reference or f not in spec:
            continue
        ours, ref = _normalize(spec[f]), _normalize(reference[f])
        if isinstance(ref, str) and ("not stated in key" in ref or "not in key" in ref or ref == ""):
            continue  # the reference does not answer this field; not comparable
        ok = _values_match(ours, ref, f)
        n_cmp += 1
        n_match += int(ok)
        fields[f] = dict(match=ok, critical=f in CRITICAL_FIELDS,
                         ours=spec[f]["value"] if isinstance(spec[f], dict) else spec[f],
                         reference=reference[f]["value"] if isinstance(reference[f], dict) else reference[f])
        if not ok:
            (crit if f in CRITICAL_FIELDS else minor).append(f)
    return dict(fields=fields, n_compared=n_cmp, n_match=n_match,
                agreement_rate=(n_match / n_cmp) if n_cmp else None,
                critical_errors=crit, minor_errors=minor)


def score_interpretation(run_dir, reference_spec_path: Path | None = None, write: bool = True) -> dict:
    run_dir = Path(run_dir)
    consensus = _load(run_dir / RunFiles.spec_consensus, {}) or {}
    spec = consensus.get("spec") or {}
    agreement = consensus.get("agreement") or {}
    validation = _load(run_dir / VALIDATION_FILE, {}) or {}
    adaptation = _load(run_dir / RunFiles.adaptation_log, []) or []

    evidence = validation.get("evidence") or {}
    invalid = sorted(f for f, e in evidence.items() if not (e or {}).get("valid", False))
    check_pass = validation.get("check_pass") or []
    n_dis = validation.get("n_check_disagreements")
    if n_dis is None:
        n_dis = sum(1 for c in check_pass if not (c or {}).get("agrees", True))

    statuses = [a.get("status") for a in adaptation]
    out = dict(
        n_runs=consensus.get("n_runs"),
        mean_field_agreement=(sum(agreement.values()) / len(agreement)) if agreement else None,
        min_field_agreement=min(agreement.values()) if agreement else None,
        field_agreement=agreement,
        evidence_valid_rate=((len(evidence) - len(invalid)) / len(evidence)) if evidence else None,
        n_evidence_checked=len(evidence),
        evidence_invalid_fields=invalid,
        n_consistency_issues=len(validation.get("consistency_issues") or []),
        n_check_disagreements=int(n_dis),
        unresolved_count=statuses.count("unresolved"),
        unresolved_details_count=len(spec.get("unresolved_details") or []),
        assumed_count=statuses.count("assumed"),
        adapted_count=statuses.count("adapted"),
        decoy_refused=(spec.get("can_implement") is False) if "can_implement" in spec else None,
        reference_label=None, preliminary=None, agreement_rate=None,
        critical_errors=None, minor_errors=None, n_critical_errors=None, n_minor_errors=None,
        n_compared=None, fields=None,
    )

    ref_path = Path(reference_spec_path) if reference_spec_path else None
    if ref_path is not None and ref_path.exists() and spec:
        reference = _unwrap_spec(json.loads(ref_path.read_text()))
        cmp = compare_to_reference(spec, reference)
        low = str(ref_path).lower()
        out.update(reference_label=ref_path.name, preliminary=("answer_key" in low or "preliminary" in low),
                   agreement_rate=cmp["agreement_rate"], critical_errors=cmp["critical_errors"],
                   minor_errors=cmp["minor_errors"], n_critical_errors=len(cmp["critical_errors"]),
                   n_minor_errors=len(cmp["minor_errors"]), n_compared=cmp["n_compared"], fields=cmp["fields"])
    elif ref_path is not None:
        out["reference_missing"] = str(ref_path.name)

    if write:
        (run_dir / RunFiles.scorecard_interpretation).write_text(json.dumps(out, indent=1))
    return out
