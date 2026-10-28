"""Pipeline orchestration: one paper -> one run folder, nine stages.

Each stage reads only files the previous stage wrote. Progress is written to
progress.json after every status change so the app can poll it. Model calls go
through research_engine.extraction.client (API) or are satisfied from files a
session agent imported (spec_raw_k.json, check_pass_k.json, narrative.json).
"""
from __future__ import annotations

import json, logging, shutil, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from . import contracts as C
from .contracts import (Progress, RunFiles, StageState, read_progress, write_progress,
                        StrategySpec, CheckPassResult, ExperimentSpec)

log = logging.getLogger("engine")


class Blocked(Exception):
    """Raised by a stage to stop the run with a recorded reason (not a failure)."""


class ProgressWriter:
    def __init__(self, run_dir: Path, run_id: str):
        self.run_dir = run_dir
        p = run_dir / RunFiles.progress
        self.p = read_progress(run_dir) if p.exists() else Progress.new(run_id)
        write_progress(run_dir, self.p)

    def update(self, n: int, status: str, detail: str = ""):
        st = self.p.stages[n - 1]
        st.status = status
        if detail:
            st.detail = detail
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if status == "running" and not st.started:
            st.started = now
        if status in ("done", "blocked", "failed"):
            st.finished = now
        write_progress(self.run_dir, self.p)

    def set(self, **kw):
        for k, v in kw.items():
            setattr(self.p, k, v)
        write_progress(self.run_dir, self.p)


def _now_id(paper_sha: str) -> str:
    return f"{paper_sha[:12]}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str))


def _stage(prog: ProgressWriter, n: int, fn: Callable[[], str]) -> None:
    """Run one stage; fn returns the one-line 'done' detail."""
    prog.update(n, "running")
    try:
        detail = fn()
        prog.update(n, "done", detail)
    except Blocked as b:
        prog.update(n, "blocked", str(b))
        raise
    except Exception as e:  # noqa: BLE001
        prog.update(n, "failed", f"{type(e).__name__}: {e}")
        (prog.run_dir / RunFiles.run_log).open("a").write(traceback.format_exc())
        raise


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------
def stage_ingest(pdf_path: Path, run_dir: Path, selected: Optional[str]) -> dict:
    from .documents.ingest import ingest
    return ingest(pdf_path, run_dir, selected)


def _load_specs(run_dir: Path) -> list[StrategySpec]:
    specs = []
    for k in (1, 2, 3):
        f = run_dir / RunFiles.spec_raw.format(k=k)
        if f.exists():
            specs.append(StrategySpec.model_validate_json(f.read_text()))
    return specs


RUN_CONFIG = "run_config.json"   # {model, eval_start, eval_end}: per-run choices made in the app or CLI


def load_run_config(run_dir: Path) -> dict:
    f = Path(run_dir) / RUN_CONFIG
    try:
        return json.loads(f.read_text()) if f.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


def stage_extract(run_dir: Path, n_runs: int, llm: str, model: Optional[str] = None) -> tuple[list[StrategySpec], dict]:
    """Return (specs, agreement). Uses existing spec_raw_k files first, then API/cache."""
    from .extraction.prompts import SYSTEM_EXTRACT, build_extract_user
    from .extraction.evidence_check import consensus
    from .documents.ingest import page_tagged_text
    specs = _load_specs(run_dir)
    if len(specs) < n_runs and llm != "files_only":
        from .extraction.client import call_structured, LLMUnavailable
        meta = json.loads((run_dir / RunFiles.paper_meta).read_text())
        text = json.loads((run_dir / RunFiles.paper_text).read_text())
        user = build_extract_user(page_tagged_text(text), meta.get("selected_experiment"))
        for k in range(len(specs) + 1, n_runs + 1):
            try:
                # distinct cache entries per k: append a run marker the model ignores
                obj, usage = call_structured(SYSTEM_EXTRACT, user + f"\n<!-- extraction run {k} -->",
                                             StrategySpec, tag=f"extract_{k}", run_dir=run_dir, model=model)
            except LLMUnavailable as e:
                if specs:
                    break
                raise Blocked(f"LLM unavailable and no extraction on file: {e}") from e
            (run_dir / RunFiles.spec_raw.format(k=k)).write_text(obj.model_dump_json(indent=1))
            specs.append(obj)
    if not specs:
        raise Blocked("no extraction available (run the session extractor or provide an API key)")
    spec, agreement = consensus(specs)
    (run_dir / RunFiles.spec_consensus).write_text(json.dumps(
        {"spec": json.loads(spec.model_dump_json()), "agreement": agreement, "n_runs": len(specs)}, indent=1))
    return specs, agreement


def stage_validate(run_dir: Path, specs: list[StrategySpec], llm: str, model: Optional[str] = None,
                   eval_start: Optional[str] = None, eval_end: Optional[str] = None) -> ExperimentSpec:
    from .extraction.evidence_check import check_evidence
    from .validation.consistency import check_consistency
    from .validation.adapt import adapt
    meta = json.loads((run_dir / RunFiles.paper_meta).read_text())
    text = json.loads((run_dir / RunFiles.paper_text).read_text())
    cons = json.loads((run_dir / RunFiles.spec_consensus).read_text())
    spec = StrategySpec.model_validate(cons["spec"])
    evidence = check_evidence(spec, text)
    checks: list[CheckPassResult] = []
    for k in range(1, len(specs) + 1):
        f = run_dir / RunFiles.check_pass.format(k=k)
        if f.exists():
            checks.append(CheckPassResult.model_validate_json(f.read_text()))
        elif llm != "files_only":
            try:
                from .extraction.client import call_structured, LLMUnavailable
                from .extraction.prompts import SYSTEM_CHECK, build_check_user
                from .documents.ingest import page_tagged_text
                obj, _ = call_structured(SYSTEM_CHECK, build_check_user(specs[k - 1], page_tagged_text(text)),
                                         CheckPassResult, tag=f"check_{k}", run_dir=run_dir, model=model)
                f.write_text(obj.model_dump_json(indent=1))
                checks.append(obj)
            except Exception as e:  # LLMUnavailable or other: check pass is advisory
                log.warning("check pass %d unavailable: %s", k, e)
    issues = check_consistency(spec)
    exp = adapt(spec, run_id=run_dir.name, paper_sha256=meta["sha256"],
                selected_experiment=meta.get("selected_experiment"),
                consistency_issues=issues, check_results=checks, evidence=evidence)
    if eval_start or eval_end:                      # window chosen for this run (app / CLI) overrides the policy
        exp = apply_window(exp, eval_start, eval_end)
    (run_dir / RunFiles.experiment_spec).write_text(exp.model_dump_json(indent=1))
    _write_json(run_dir / RunFiles.adaptation_log, [a.model_dump() for a in exp.adaptations])
    _write_json(run_dir / RunFiles.validation, {
        "evidence": evidence, "consistency_issues": issues,
        "check_pass": [c.model_dump() for c in checks],
        "n_check_disagreements": sum(len(c.items) for c in checks)})
    if exp.status == "blocked":
        raise Blocked("; ".join(exp.blocked_reasons) or "blocked by validation")
    return exp


def apply_window(exp: ExperimentSpec, eval_start: Optional[str], eval_end: Optional[str]) -> ExperimentSpec:
    """Replace the evaluation window and rewrite the sample_period adaptation entry to say so."""
    start = str(eval_start or exp.eval_start)[:10]
    end = str(eval_end or exp.eval_end)[:10]
    if start < exp.warmup_start:
        raise Blocked(f"backtest start {start} is before the data warm-up {exp.warmup_start}")
    if end <= start:
        raise Blocked(f"backtest window is empty: {start} to {end}")
    exp = exp.model_copy(update={"eval_start": start, "eval_end": end})
    for a in exp.adaptations:
        if a.field == "sample_period":
            a.our_value = f"{start}..{end} (warm-up from {exp.warmup_start})"
            a.reason = "window chosen for this run; warm-up uses all earlier snapshot data"
    return exp


def stage_data(run_dir: Path, exp: ExperimentSpec):
    from .data.loaders import DataContext
    ctx = DataContext.load()
    month_ends = ctx.month_ends[(ctx.month_ends >= pd.Timestamp(exp.eval_start)) & (ctx.month_ends <= pd.Timestamp(exp.eval_end))]
    counts = {str(d.date()): int(len(ctx.eligible(d, exp.min_history_days, exp.price_floor))) for d in month_ends[::12]}
    manifest = json.loads((C.SNAPSHOT_DIR / "manifest.json").read_text())
    _write_json(run_dir / "data_check.json", {
        "snapshot": manifest["snapshot"], "n_universe": manifest["n_universe"],
        "excluded": manifest.get("excluded", []), "eligible_by_year": counts,
        "survivor_universe": True, "calendar_days": int(len(ctx.calendar))})
    return ctx, counts


def stage_construct(run_dir: Path, ctx, exp: ExperimentSpec):
    from .strategies.portfolios import build_targets
    res = build_targets(ctx, exp)
    res.signals_long.to_parquet(run_dir / RunFiles.signals)
    _write_json(run_dir / RunFiles.plan, {"signal_ops": [o.model_dump() for o in exp.signal_ops],
                                          "selection": exp.selection, "holding_months": exp.holding_months,
                                          "skipped_months": [str(d) for d in getattr(res, "skipped_months", [])]})
    return res


def stage_backtest(run_dir: Path, ctx, exp: ExperimentSpec, res):
    from .strategies.dev_run import backtest_all
    return backtest_all(ctx, exp, res, run_dir)


def stage_evaluate(run_dir: Path, reference_spec: Optional[Path]):
    from .evaluation.strategy import score_strategy
    from .evaluation.interpretation import score_interpretation
    s = score_strategy(run_dir)
    i = score_interpretation(run_dir, reference_spec)
    return s, i


def stage_economics(run_dir: Path, exp: ExperimentSpec):
    from .economics.fund import run_economics
    returns = pd.read_parquet(run_dir / RunFiles.returns)
    usage_f = run_dir / RunFiles.usage
    llm_cost = 0.0
    if usage_f.exists():
        for u in json.loads(usage_f.read_text()):
            if str(u.get('call', '')).startswith('narrative'):
                continue  # the narrative is written after the economics; its cost cannot be cited by itself
            llm_cost += u.get("input_tokens", 0) * 2 / 1e6 + u.get("output_tokens", 0) * 10 / 1e6
    summary, paths = run_economics(returns, exp.fund, exp.research_cost, exp.monte_carlo,
                                   llm_cost_usd=llm_cost, seed=exp.monte_carlo["seed"])
    paths.to_parquet(run_dir / RunFiles.montecarlo)
    _write_json(run_dir / RunFiles.economics, summary)
    return summary


def stage_dossier(run_dir: Path, llm: str, model: Optional[str] = None) -> dict:
    from .reporting.results import assemble_results
    from .reporting.rubric import recommend
    from .reporting.narrative import write_narrative
    from .reporting.render import render_dossier, render_index
    results = assemble_results(run_dir)
    rec = recommend(results)
    results["rubric"] = rec["rubric"]; results["recommendation"] = rec["recommendation"]
    results["recommendation_reasons"] = rec["reasons"]
    _write_json(run_dir / RunFiles.results, results)
    narrative = write_narrative(run_dir, results, llm=llm, model=model)   # handles API / files / fallback
    render_dossier(run_dir, results, narrative)
    render_index(C.RUNS_DIR)
    return {"recommendation": results["recommendation"], "summary": narrative.get("summary", "")}


# ----------------------------------------------------------------------------
# Extraction-only runs (dropped PDFs)
# ----------------------------------------------------------------------------
def run_extract_only(run_dir: Path, prog: ProgressWriter, specs: list[StrategySpec], llm: str,
                     model: Optional[str]) -> None:
    from .extraction.summary import write_summary
    cons = json.loads((run_dir / RunFiles.spec_consensus).read_text())
    spec = StrategySpec.model_validate(cons["spec"])
    verdict = {"text": ""}

    def s3():
        try:
            exp = stage_validate(run_dir, specs, llm, model)
            prog.set(strategy_slug=exp.strategy_slug)
            verdict["text"] = (f"The engine could run this as '{exp.strategy_slug}' with {exp.n_assumed} assumed and "
                               f"{exp.n_adapted} adapted details.")
            return f"engine could run it as '{exp.strategy_slug}' · {exp.n_assumed} assumed · {exp.n_adapted} adapted"
        except Blocked as b:
            verdict["text"] = f"The engine cannot run this strategy: {b}"
            return f"not implementable by the engine: {b}"
    _stage(prog, 3, s3)
    for n in range(4, 9):
        prog.update(n, "skipped", "not run: dropped PDF, extraction only")
    prog.p.stages[8].name = "Strategy summary"

    def s9():
        out = write_summary(run_dir, spec, verdict["text"], llm=llm, model=model)
        prog.set(recommendation="extraction only", summary=out["summary"])
        return f"written ({out.get('source', '?')})"
    _stage(prog, 9, s9)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run_pipeline(pdf_path: Path, run_dir: Optional[Path] = None, selected_experiment: Optional[str] = None,
                 llm: str = "api", n_extractions: int = 3, reference_spec: Optional[Path] = None,
                 start_stage: int = 1, model: Optional[str] = None,
                 eval_start: Optional[str] = None, eval_end: Optional[str] = None,
                 extract_only: bool = False) -> Path:
    """llm: 'api' (call Claude, cache first) | 'files_only' (never call; use spec_raw/check/narrative files).
    model / eval_start / eval_end: per-run choices; None = contracts defaults. They are written to
    run_config.json and reused on resume when not given again.
    extract_only: stop after extraction (dropped PDFs): stage 3 only reports whether the engine could run the
    strategy, stages 4-8 are skipped, and stage 9 writes a model summary of the strategy instead of a dossier."""
    pdf_path = Path(pdf_path)
    if run_dir is None:
        import hashlib
        sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        run_dir = C.RUNS_DIR / _now_id(sha)
    run_dir.mkdir(parents=True, exist_ok=True)
    prog = ProgressWriter(run_dir, run_dir.name)
    cfg = load_run_config(run_dir)
    model = model or cfg.get("model") or None
    eval_start = eval_start or cfg.get("eval_start") or None
    eval_end = eval_end or cfg.get("eval_end") or None
    _write_json(run_dir / RUN_CONFIG, {"model": model or C.MODEL, "eval_start": eval_start or C.EVAL_START,
                                       "eval_end": eval_end or C.EVAL_END, "warmup_start": C.WARMUP_START,
                                       "custom_window": bool(eval_start or eval_end), "extract_only": bool(extract_only)})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(run_dir / RunFiles.run_log), logging.StreamHandler()])
    exp = None; ctx = None; res = None; specs = []
    try:
        if start_stage <= 1:
            def s1():
                meta = stage_ingest(pdf_path, run_dir, selected_experiment)
                prog.set(paper_title=meta.get("title") or meta.get("title_guess", ""))
                return f"{meta['n_pages']} pages · {meta['sha256'][:8]}"
            _stage(prog, 1, s1)
        if start_stage > 1 and (run_dir / RunFiles.paper_meta).exists():
            meta = json.loads((run_dir / RunFiles.paper_meta).read_text())
            prog.set(paper_title=meta.get("title") or meta.get("title_guess", ""))
            prog.update(1, "done", f"{meta.get('n_pages', '?')} pages · {meta.get('sha256', '')[:8]}")
        if start_stage <= 2:
            def s2():
                nonlocal specs
                specs, agreement = stage_extract(run_dir, n_extractions, llm, model)
                mean_agr = sum(agreement.values()) / max(len(agreement), 1)
                return f"{len(specs)} run(s) · fields agree {mean_agr:.0%}"
            _stage(prog, 2, s2)
        else:
            specs = _load_specs(run_dir)
        if extract_only:
            run_extract_only(run_dir, prog, specs, llm, model)
            return run_dir
        if start_stage <= 3:
            def s3():
                nonlocal exp
                exp = stage_validate(run_dir, specs, llm, model, eval_start, eval_end)
                prog.set(strategy_slug=exp.strategy_slug)
                return f"ready · {exp.n_assumed} assumed · {exp.n_adapted} adapted"
            _stage(prog, 3, s3)
        else:
            exp = ExperimentSpec.model_validate_json((run_dir / RunFiles.experiment_spec).read_text())
            prog.set(strategy_slug=exp.strategy_slug)
        if start_stage <= 4:
            def s4():
                nonlocal ctx
                ctx, counts = stage_data(run_dir, exp)
                last = list(counts.values())[-1] if counts else 0
                return f"{last} eligible names · {len(ctx.calendar)} trading days"
            _stage(prog, 4, s4)
        elif start_stage <= 6:
            from .data.loaders import DataContext
            ctx = DataContext.load()
        if start_stage <= 5:
            def s5():
                nonlocal res
                res = stage_construct(run_dir, ctx, exp)
                n_months = res.signals_long["date"].nunique()
                avg = res.signals_long[res.signals_long["selected"]].groupby("date").size().mean()
                return f"{n_months} months · {avg:.0f} names/month"
            _stage(prog, 5, s5)
        elif start_stage <= 6:
            res = stage_construct(run_dir, ctx, exp)  # cheap; needed by the backtest stage
        if start_stage <= 6:
            def s6():
                out = stage_backtest(run_dir, ctx, exp, res)
                return out if isinstance(out, str) else "done"
            _stage(prog, 6, s6)
        if start_stage <= 7:
            def s7():
                s, i = stage_evaluate(run_dir, reference_spec)
                perf = s.get("performance", {}).get("strategy_net", {})
                return f"net CAGR {perf.get('cagr', float('nan')):.1%} · alpha t {s.get('alpha', {}).get('alpha_t', float('nan')):.2f}"
            _stage(prog, 7, s7)
        if start_stage <= 8:
            def s8():
                summ = stage_economics(run_dir, exp)
                return f"{exp.monte_carlo['n_paths']} paths · P(NPV>0) {summ.get('frac_npv_positive', float('nan')):.0%}"
            _stage(prog, 8, s8)
        def s9():
            out = stage_dossier(run_dir, llm, model)
            prog.set(recommendation=out["recommendation"], summary=out["summary"])
            return out["recommendation"]
        _stage(prog, 9, s9)
    except Blocked as b:
        # Blocked is a legitimate outcome: write a minimal dossier with "unable to evaluate"
        try:
            from .reporting.render import render_blocked, render_index
            summary = render_blocked(run_dir, str(b))
            render_index(C.RUNS_DIR)
        except Exception:  # reporting not built yet
            summary = f"Unable to evaluate. {b}"
        prog.set(recommendation="unable to evaluate", summary=summary)
    except Exception as e:  # noqa: BLE001
        prog.set(error=f"{type(e).__name__}: {e}")
        raise
    return run_dir
