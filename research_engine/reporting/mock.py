"""Synthetic run folder for building and testing the reporting stage.

`build_mock_run(run_dir)` writes every file stages 1-8 would produce, with
plausible synthetic values (MAX-style strategy, labeled MOCK in the title).
Nothing here is a real result. Run as a script to (re)build runs/mock_report:

    .venv/bin/python -m research_engine.reporting.mock
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_engine import contracts as C
from research_engine.contracts import RunFiles

MOCK_TITLE = "[MOCK] Maxing Out: Stocks as Lotteries and the Cross-Section of Expected Returns"


def _ev(page: int, quote: str) -> dict:
    return {"page": page, "quote": quote}


def mock_spec() -> dict:
    """A StrategySpec-shaped dict (synthetic paraphrases, not real quotes)."""
    return {
        "paper_title": MOCK_TITLE,
        "authors": ["Mock Author A", "Mock Author B", "Mock Author C"],
        "year": 2011,
        "selected_experiment": "Table 1, univariate decile portfolios sorted on MAX",
        "can_implement": True,
        "cannot_implement_reasons": [],
        "economic_rationale": {"value": "investors overpay for lottery-like stocks with extreme positive daily returns",
                               "evidence": _ev(2, "(mock) investors may pay a premium for lottery-like payoffs")},
        "universe_and_filters": {"value": "all NYSE/AMEX/NASDAQ common stocks",
                                 "evidence": _ev(6, "(mock) the sample covers common stocks on the three exchanges")},
        "required_inputs": {"value": ["daily returns"],
                            "evidence": _ev(6, "(mock) daily stock returns from CRSP")},
        "signal_formula_words": {"value": "maximum daily return within the previous month",
                                 "evidence": _ev(7, "(mock) MAX is the maximum daily return over the past month")},
        "signal_formula_symbols": {"value": "MAX_{i,t} = max(R_{i,d}), d = 1..D_t",
                                   "evidence": _ev(7, "(mock) MAX_{i,t} = max(R_{i,d})")},
        "formation_window": {"value": "1 month of daily data",
                             "evidence": _ev(7, "(mock) over the previous month")},
        "skip_window": {"value": "none", "evidence": _ev(8, "(mock) portfolios are formed at month end")},
        "holding_period": {"value": "1 month", "evidence": _ev(8, "(mock) held for the next month")},
        "rebalance_frequency": {"value": "monthly", "evidence": _ev(8, "(mock) rebalanced every month")},
        "sort_rule": "deciles",
        "sort_rule_evidence": _ev(8, "(mock) stocks are sorted into deciles"),
        "breakpoints": {"value": "equal-count deciles", "evidence": _ev(8, "(mock) decile breakpoints")},
        "favored_group": {"value": "lowest MAX decile", "evidence": _ev(9, "(mock) low-MAX stocks earn higher returns")},
        "disfavored_group": {"value": "highest MAX decile", "evidence": _ev(9, "(mock) high-MAX stocks underperform")},
        "weighting": "value",
        "weighting_evidence": _ev(9, "(mock) value-weighted decile returns"),
        "expected_sign": "favored_group_higher",
        "expected_sign_evidence": _ev(9, "(mock) the return spread between high and low MAX is negative"),
        "headline_result": {"value": "high-minus-low MAX spread of about -1% per month",
                            "evidence": _ev(10, "(mock) the difference is about -1 percent per month")},
        "unresolved_details": ["minimum price filter", "minimum days of data in the formation month"],
    }


def mock_experiment_spec(run_id: str, sha: str) -> dict:
    adaptations = [
        {"field": "weighting", "paper_value": "value", "our_value": "equal", "status": "adapted",
         "reason": "engine supports equal weighting only"},
        {"field": "strategy_portfolio", "paper_value": "high-minus-low spread", "our_value": "long lowest MAX decile",
         "status": "adapted", "reason": "long-only adaptation; no short leg"},
        {"field": "universe_and_filters", "paper_value": "all NYSE/AMEX/NASDAQ common stocks",
         "our_value": "415-name survivor universe", "status": "adapted", "reason": "snapshot universe"},
        {"field": "price_floor", "paper_value": None, "our_value": "none", "status": "assumed",
         "reason": "paper silent; default no price filter"},
        {"field": "min_formation_days", "paper_value": None, "our_value": "15", "status": "assumed",
         "reason": "paper silent; default 15 trading days"},
        {"field": "signal_formula_words", "paper_value": "maximum daily return within the previous month",
         "our_value": "rolling_max_daily_return(window_months=1)", "status": "stated", "reason": "maps to registry op"},
        {"field": "holding_period", "paper_value": "1 month", "our_value": "1", "status": "stated", "reason": "as stated"},
    ]
    exp = C.ExperimentSpec(
        run_id=run_id, paper_sha256=sha, strategy_slug="max", status="ready",
        signal_ops=[C.SignalOp(op="rolling_max_daily_return", params={"window_months": 1, "min_days": 15})],
        selection={"sort": "deciles", "n_groups": 10, "percent": None, "favored": "lowest", "double_sort": None},
        expected_sign="favored_group_higher",
        adaptations=[C.AdaptationEntry(**a) for a in adaptations],
        n_assumed=sum(a["status"] == "assumed" for a in adaptations),
        n_adapted=sum(a["status"] == "adapted" for a in adaptations),
    )
    return json.loads(exp.model_dump_json())


def _stats(r: pd.Series) -> dict:
    n = len(r)
    growth = float((1 + r).prod())
    cagr = growth ** (252 / n) - 1
    vol = float(r.std() * np.sqrt(252))
    wealth = (1 + r).cumprod()
    mdd = float((wealth / wealth.cummax() - 1).min())
    return {"cagr": cagr, "ann_vol": vol, "sharpe": cagr / vol if vol else None,
            "max_drawdown": mdd, "cum_return": growth - 1}


def _alpha_t(y: pd.Series, x: pd.Series) -> tuple[float, float, float]:
    X = np.column_stack([np.ones(len(x)), x.values])
    b, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    resid = y.values - X @ b
    s2 = resid @ resid / (len(y) - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return float(b[0] * 252), float(b[0] / se[0]), float(b[1])


def build_mock_run(run_dir: Path, seed: int = 7) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    w = lambda name, obj: (run_dir / name).write_text(json.dumps(obj, indent=1, default=str))
    sha = "ab12cd34ef56" + "0" * 52
    run_id = run_dir.name

    w(RunFiles.paper_meta, {"sha256": sha, "source_path": "papers/pdf/mock_max.pdf", "n_pages": 42,
                            "title": MOCK_TITLE, "selected_experiment": "Table 1"})
    spec = mock_spec()
    agreement = {f: 1.0 for f in C.MATERIAL_FIELDS}
    agreement.update({"breakpoints": 2 / 3, "weighting": 2 / 3})
    w(RunFiles.spec_consensus, {"spec": spec, "agreement": agreement, "n_runs": 3})
    w(RunFiles.spec_raw.format(k=1), spec)
    evidence = {f: {"valid": True, "page": (spec[f]["evidence"]["page"] if isinstance(spec[f], dict)
                                            else spec[f + "_evidence"]["page"])} for f in C.MATERIAL_FIELDS}
    evidence["headline_result"]["valid"] = False
    w("validation.json", {"evidence": evidence, "consistency_issues": [],
                          "check_pass": [{"agrees": True, "items": []},
                                         {"agrees": False, "items": [{"field": "breakpoints",
                                                                      "issue": "paper uses NYSE breakpoints",
                                                                      "contradicting_quote": "(mock) NYSE breakpoints",
                                                                      "page": 8}]},
                                         {"agrees": True, "items": []}],
                          "n_check_disagreements": 1})
    exp = mock_experiment_spec(run_id, sha)
    w(RunFiles.experiment_spec, exp)
    w(RunFiles.adaptation_log, exp["adaptations"])
    years = range(2015, 2026)
    w("data_check.json", {"snapshot": C.SNAPSHOT_DIR.name, "n_universe": 415,
                          "excluded": [f"MOCK{i}" for i in range(19)],
                          "eligible_by_year": {f"{y}-01-30": 340 + 6 * (y - 2015) for y in years},
                          "survivor_universe": True, "calendar_days": 3521})

    # daily returns 2015-2025
    dates = pd.bdate_range(C.EVAL_START, C.EVAL_END)
    n = len(dates)
    mkt = rng.normal(0.0005, 0.0105, n)
    ew = mkt * 1.02 + rng.normal(0.00002, 0.003, n)
    groups = {}
    for k in range(1, 11):
        tilt = 0.00016 - 0.00003 * (k - 1)
        beta = 0.75 + 0.06 * (k - 1)
        groups[f"group_{k}"] = mkt * beta + tilt + rng.normal(0, 0.004 + 0.0007 * k, n)
    gross = groups["group_1"] + 0.00001
    net = gross - 0.00003
    ret = pd.DataFrame({"strategy_gross": gross, "strategy_net": net, "spy": mkt, "ew_universe": ew,
                        **groups}, index=dates)
    ret.index.name = "date"
    ret.to_parquet(run_dir / RunFiles.returns)

    # portfolio value
    nv = 1e8 * (1 + ret["strategy_net"]).cumprod()
    gv = 1e8 * (1 + ret["strategy_gross"]).cumprod()
    first_days = ret.groupby(ret.index.to_period("M")).head(1).index
    pv = pd.DataFrame({"gross_value": gv, "net_value": nv, "cash": 0.0,
                       "n_holdings": rng.integers(32, 40, n), "turnover": np.nan}, index=dates)
    pv.loc[first_days, "turnover"] = rng.uniform(0.35, 0.6, len(first_days))
    pv.to_parquet(run_dir / RunFiles.portfolio_value)

    # signals (month-ends, 350 names, deciles, lowest selected)
    month_ends = ret.groupby(ret.index.to_period("M")).tail(1).index
    skipped = [month_ends[5]]
    rows = []
    tickers = [f"T{i:03d}" for i in range(350)]
    for d in month_ends:
        if d in skipped:
            continue
        sig = rng.gamma(2.0, 0.02, len(tickers))
        grp = pd.qcut(sig, 10, labels=False) + 1
        for t, s, g in zip(tickers, sig, grp):
            rows.append((d, t, float(s), int(g), bool(g == 1), (1 / 35) if g == 1 else 0.0, d))
    sig = pd.DataFrame(rows, columns=["date", "ticker", "signal", "group", "selected", "weight", "cohort"])
    sig.to_parquet(run_dir / RunFiles.signals)
    w(RunFiles.plan, {"signal_ops": exp["signal_ops"], "selection": exp["selection"], "holding_months": 1,
                      "skipped_months": [str(d) for d in skipped]})

    # scorecard_strategy
    perf = {c: _stats(ret[c]) for c in ("strategy_net", "strategy_gross", "spy", "ew_universe")}
    a_ann, a_t, beta = _alpha_t(ret["strategy_net"] - ret["ew_universe"], ret["spy"])
    capm_a, capm_t, capm_b = _alpha_t(ret["strategy_net"], ret["spy"])

    def cost_entry(bps):
        r = ret["strategy_gross"] - bps * 1e-4 * 0.45 / 21
        s = _stats(r)
        aa, tt, _ = _alpha_t(r - ret["ew_universe"], ret["spy"])
        return {"cagr": s["cagr"], "sharpe": s["sharpe"], "alpha_t": tt,
                "excess_cagr_vs_ew": s["cagr"] - perf["ew_universe"]["cagr"]}

    def sub(lo, hi):
        m = (ret.index >= lo) & (ret.index < hi)
        s, e = _stats(ret.loc[m, "strategy_net"]), _stats(ret.loc[m, "ew_universe"])
        _, tt, _ = _alpha_t(ret.loc[m, "strategy_net"] - ret.loc[m, "ew_universe"], ret.loc[m, "spy"])
        return {"cagr": s["cagr"], "ew_cagr": e["cagr"], "sharpe": s["sharpe"], "alpha_t": tt,
                "excess_cagr_vs_ew": s["cagr"] - e["cagr"]}

    gtable = [{"group": k, **{kk: v for kk, v in _stats(ret[f"group_{k}"]).items() if kk in ("cagr", "sharpe", "ann_vol")}}
              for k in range(1, 11)]
    w(RunFiles.scorecard_strategy, {
        "performance": perf,
        "relative": {"vs_spy": {"excess_cagr": perf["strategy_net"]["cagr"] - perf["spy"]["cagr"], "nw_t": 0.9},
                     "vs_ew": {"excess_cagr": perf["strategy_net"]["cagr"] - perf["ew_universe"]["cagr"], "nw_t": a_t}},
        "alpha": {"capm_alpha_ann": capm_a, "capm_beta": capm_b, "capm_t": capm_t,
                  "ff4_alpha_ann": a_ann, "alpha_t": a_t, "nw_lag": 1,
                  "loadings": {"mkt_rf": 0.81, "smb": 0.12, "hml": 0.18, "mom": 0.05}},
        "spanning": {"alpha_ann": a_ann * 0.9, "alpha_t": a_t * 0.9, "r2": 0.41},
        "turnover": {"mean_monthly": float(pv["turnover"].mean()), "annual": float(pv["turnover"].mean() * 12)},
        "costs": {"annual_cost_drag": perf["strategy_gross"]["cagr"] - perf["strategy_net"]["cagr"], "cost_bps": 10},
        "holdings": {"mean_names": 35.0, "min_names": 32, "max_names": 39, "mean_top10_weight": 0.2857},
        "capacity": {"median_pct_adv": 0.0021, "p95_pct_adv": 0.0118, "aum": 1e8},
        "robustness": {
            "cost_grid": {str(b): cost_entry(b) for b in C.COST_BPS_GRID},
            "subperiods": {"2015-2019": sub("2015-01-01", C.SUBPERIOD_SPLIT),
                           "2020-2025": sub(C.SUBPERIOD_SPLIT, "2026-01-01")},
            "groups": {"table": gtable, "rho": -0.92, "rho_p": 0.0002, "rho_in_expected_direction": True,
                       "favored_group": 1},
        },
    })
    w(RunFiles.scorecard_interpretation, {
        "n_runs": 3, "mean_field_agreement": float(np.mean(list(agreement.values()))),
        "evidence_valid_rate": 15 / 16, "n_check_disagreements": 1, "unresolved_count": 2,
        "assumed_count": exp["n_assumed"], "adapted_count": exp["n_adapted"], "decoy_refused": None,
        "agreement_rate": 0.875, "critical_errors": 0, "minor_errors": 2,
        "reference_label": "preliminary answer key", "preliminary": True})

    # Monte Carlo paths and economics
    P = 1000
    ew_w = np.exp(rng.normal(0.35, 0.22, P))
    st_w = ew_w * np.exp(rng.normal(0.03, 0.08, P))
    spy_w = ew_w * np.exp(rng.normal(0.01, 0.06, P))
    fees_ew = 0.02 * 5 * 1e8 * ew_w ** 0.5 + 2e6 * np.maximum(ew_w - 1.28, 0) * 10
    fees_st = fees_ew + rng.normal(9e5, 1.6e6, P)
    inc = (fees_st - fees_ew) * 0.8
    research = 80 * 150 + 10 * 150 * 5 + 1.37
    npv = inc - research
    paths = pd.DataFrame({
        "end_wealth_strategy": st_w, "end_wealth_ew": ew_w, "end_wealth_spy": spy_w,
        "maxdd_strategy": -np.abs(rng.normal(0.22, 0.07, P)), "fees_strategy": fees_st, "fees_ew": fees_ew,
        "incremental_fees_pv": inc, "npv": npv, "beats_spy": st_w > spy_w, "beats_ew": st_w > ew_w,
        "npv_demeaned": npv - 7e5, "incremental_fees_pv_demeaned": inc - 7e5})
    paths.index.name = "path"
    paths.to_parquet(run_dir / RunFiles.montecarlo)
    qcols = ["end_wealth_strategy", "end_wealth_ew", "end_wealth_spy", "maxdd_strategy", "fees_strategy",
             "fees_ew", "incremental_fees_pv", "npv", "npv_demeaned", "incremental_fees_pv_demeaned"]
    exp_inc = float(inc.mean())
    w(RunFiles.economics, {
        "quantiles": {c: {f"p{q:02d}": float(np.percentile(paths[c], q)) for q in (5, 25, 50, 75, 95)} for c in qcols},
        "frac_beats_spy": float(paths["beats_spy"].mean()), "frac_beats_ew": float(paths["beats_ew"].mean()),
        "frac_npv_positive": float((npv > 0).mean()), "frac_npv_positive_demeaned": float((npv - 7e5 > 0).mean()),
        "expected_incremental_fees_pv": exp_inc, "expected_npv": float(npv.mean()),
        "breakeven_aum": research / (exp_inc / 1e8) if exp_inc > 0 else None,
        "research_cost_total": research, "llm_cost_usd": 1.37,
        "block_sensitivity": {"63": 0.61, "126": float((npv > 0).mean()), "252": 0.66},
        "n_paths": P, "horizon_days": 1260, "mean_block_days": 126, "benchmark": "ew_universe",
        "assumptions": ["2/20 with a 5% hard hurdle on a long-only equity strategy is an assumption (mock)",
                        "incremental fees are measured versus an equal-weight universe fund, never versus SPY",
                        "break-even AUM is reported, not headlined"]})
    w(RunFiles.usage, [
        {"call": f"extract_{k}", "model": C.MODEL, "input_tokens": 61000, "output_tokens": 3200, "cached": k > 1}
        for k in (1, 2, 3)] + [
        {"call": f"check_{k}", "model": C.MODEL, "input_tokens": 64000, "output_tokens": 900, "cached": False}
        for k in (1, 2, 3)])
    prog = C.Progress.new(run_id)
    prog.paper_title = MOCK_TITLE
    prog.strategy_slug = "max"
    for st in prog.stages:
        st.status = "done" if st.n < 9 else "pending"
    C.write_progress(run_dir, prog)
    return run_dir


def build_mock_blocked(run_dir: Path) -> Path:
    """A folder with only paper_meta + spec files (the blocked-run case)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = mock_spec()
    spec.update(paper_title="[MOCK] Decoy: The Other Side of Value", can_implement=False,
                cannot_implement_reasons=["requires accounting data (gross profitability)"])
    (run_dir / RunFiles.paper_meta).write_text(json.dumps(
        {"sha256": "f" * 64, "source_path": "papers/pdf/mock_decoy.pdf", "n_pages": 30,
         "title": spec["paper_title"], "selected_experiment": None}))
    (run_dir / RunFiles.spec_raw.format(k=1)).write_text(json.dumps(spec))
    (run_dir / RunFiles.spec_consensus).write_text(json.dumps(
        {"spec": spec, "agreement": {f: 1.0 for f in C.MATERIAL_FIELDS}, "n_runs": 1}))
    return run_dir


if __name__ == "__main__":  # pragma: no cover
    out = build_mock_run(C.RUNS_DIR / "mock_report")
    print(out)
