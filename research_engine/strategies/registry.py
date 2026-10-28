"""Registry of time-series signal operations.

Every op has the signature ``op(ctx, dates, **params) -> DataFrame`` indexed by
``dates`` (formation month-ends), columns = ctx.tickers, and may use only data
with index <= each date. Ops live in ``research_engine/strategies/ops/`` and
register themselves on import.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

_OPS: dict[str, Callable[..., pd.DataFrame]] = {}
_LOADED = False


def register(name: str):
    def deco(fn):
        if name in _OPS and _OPS[name] is not fn:
            raise ValueError(f"op {name!r} already registered")
        _OPS[name] = fn
        fn.op_name = name
        return fn
    return deco


def _ensure_loaded() -> None:
    global _LOADED
    if not _LOADED:
        import research_engine.strategies.ops  # noqa: F401  (registers on import)
        _LOADED = True


def get(name: str) -> Callable[..., pd.DataFrame]:
    _ensure_loaded()
    try:
        return _OPS[name]
    except KeyError:
        raise KeyError(f"unknown signal op {name!r}; known: {sorted(_OPS)}") from None


def list_ops() -> list[str]:
    _ensure_loaded()
    return sorted(_OPS)


def compute(ctx, dates, op: str, params: dict | None = None) -> pd.DataFrame:
    """Run one registered op and check its output shape."""
    dates = pd.DatetimeIndex(dates)
    out = get(op)(ctx, dates, **(params or {}))
    if not out.index.equals(dates):
        raise AssertionError(f"op {op} returned an index that is not `dates`")
    return out
