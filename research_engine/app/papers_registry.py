"""Known papers for the app's left column: (label, path relative to repo root, strategy slug)."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REGISTRY: list[tuple[str, str, str]] = [
    ("Bali, Cakici & Whitelaw 2011 — MAX", "papers/pdf/bali_cakici_whitelaw_2011_maxing_out_nber14804.pdf", "max"),
    ("Jegadeesh & Titman 2001 — momentum", "papers/pdf/jegadeesh_titman_2001_momentum_nber7159.pdf", "momentum"),
    ("Heston & Sadka 2008 — seasonality", "papers/pdf/heston_sadka_2008_seasonality_wp.pdf", "seasonality"),
    ("Ang, Hodrick, Xing & Zhang 2006 — low volatility", "papers/pdf/ang_hodrick_xing_zhang_2006_volatility_nber10852.pdf", "lowvol"),
    ("George & Hwang 2004 — 52-week high", "papers/pdf/george_hwang_2004_52week_high_jf.pdf", "high52"),
    ("Da, Gurun & Warachka 2014 — frog in the pan", "papers/pdf/da_gurun_warachka_2014_frog_in_the_pan_rfs.pdf", "frog"),
    ("Oil factor (Aidan) — mock paper", "papers/oil_factor_mock_paper_aidan.pdf", "oil"),
    ("Novy-Marx 2013 — profitability (decoy)", "papers/pdf/decoy_novy_marx_2013_gross_profitability_nber15940.pdf", "decoy_profitability"),
    ("Moskowitz, Ooi & Pedersen 2012 — time-series momentum (decoy)", "papers/pdf/decoy_moskowitz_ooi_pedersen_2012_time_series_momentum_jfe.pdf", "decoy_tsmom"),
]


def abs_path(rel: str) -> Path:
    return ROOT / rel


@lru_cache(maxsize=1)
def known_papers() -> dict:
    """{"sha": {sha256: label}, "file": {file_name: label}} for registry PDFs that exist (hashed once per process)."""
    sha, file = {}, {}
    for label, rel, _ in REGISTRY:
        p = abs_path(rel)
        file[p.name] = label
        try:
            sha[hashlib.sha256(p.read_bytes()).hexdigest()] = label
        except OSError:
            pass
    return {"sha": sha, "file": file}
