# Research Screening Agent

IEOR 4723 (AI-Augmented Investment Management), Columbia. Group 6.

One research paper in, one evidence-backed research dossier out. The engine
extracts the paper's trading strategy with an LLM (page-cited), validates and
adapts it by explicit policy, builds a long-only equal-weighted portfolio on a
415-name S&P 500 universe (2015-2025), backtests it with holdings-level
accounting, evaluates performance, robustness and interpretation accuracy,
simulates fund economics by block bootstrap, and writes a dossier with one of
four recommendations: pursue, investigate further, deprioritize, unable to
evaluate. Python does every number; the LLM only extracts, checks and explains.

## Run it

```bash
cd "AI_Applications_Project_philip"
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"      # once
.venv/bin/pytest -q                                              # test suite
.venv/bin/streamlit run research_engine/app/streamlit_app.py     # the app, http://localhost:8501
```

Command line, one paper end to end:

```bash
.venv/bin/python -m research_engine.cli all papers/pdf/bali_cakici_whitelaw_2011_maxing_out_nber14804.pdf
```

Add `--llm files_only` to run with no model calls (uses cached extractions).

## Model access

**First run on a new machine: the app asks for a key.** With no key on the
machine, WARREN opens on a key setup step. The default is the Anthropic key
shared in class; pick "Personal key" to use your own. The key is checked with
one tiny call (`max_tokens=5`) before anything is stored, then written to
`.env` (gitignored, mode 600) with a label (`ANTHROPIC_KEY_LABEL=class|personal`)
that each run records in its `run_config.json`. "Keep for this session only"
leaves the disk untouched. You can also continue without a key and open the
cached runs. The Session panel (bottom of the right column) replaces or removes
the key later. `.env.example` shows the two lines if you prefer to write the
file yourself. Never paste the key into a notebook, slide, chat or commit.

Put `ANTHROPIC_API_KEY=...` in `.env` (gitignored). The model is
`claude-sonnet-5-5` (`research_engine/contracts.py: MODEL`). Every model
response is cached under `cache/llm/` keyed by paper, prompt, schema and model,
so the nine demo papers run with no key. If the key is missing or rejected the
pipeline uses cached responses and, failing that, blocks the extraction stage
with a clear message. The build night's key was rejected by the API, so the
demo extractions were produced by the same prompts executed in a Claude
session and imported into the cache with `source: "session"`; with a working
key the same prompts run through the API unchanged.

## What is where

```
research_engine/
  contracts.py      schemas (StrategySpec, ExperimentSpec, Progress), constants, file names
  pipeline.py       the nine stages; writes progress.json after every change
  cli.py            all | resume | index | export-prompts | app
  documents/        PDF -> page-tagged text
  extraction/       prompts, Claude client with cache, evidence checker, consensus, session runner
  validation/       consistency checks, defaults, adaptation policies (paper -> experiment)
  data/             snapshot builder, loaders (eligibility, patches, exclusions)
  strategies/       operation registry (ops/), construction, cohorts, dev runner
  backtest/         holdings-based engine, benchmarks
  evaluation/       strategy scorecard, interpretation scorecard, statistics
  economics/        fee engine (from the original project), bootstrap, fund economics
  reporting/        results assembly, rubric, narrative with number check, HTML dossier and index
  app/              Streamlit app
tests/              184+ tests incl. engine invariants and an independent MAX reimplementation
data/snapshot_*/    frozen Yahoo + Ken French snapshot (gitignored; manifest and quality report inside)
papers/             answer key (grading only), reference specs (preliminary), PDFs (gitignored)
runs/               one folder per run; demo_* are committed; index.html compares runs
legacy/             the original project (Aidan's optimizer, notebook, data) unchanged
BUILD_PLAN_FINAL.md the plan; BUILD_LOG.md what happened and every decision taken overnight
```

## Conventions that matter

Signal through the prior month's last close; execution at the next month's
first close; equal weights; 10 bps per dollar traded (0 and 25 reported);
one-month holding unless the paper holds longer, then overlapping monthly
cohorts; no price floors (the snapshot's prices are split-adjusted, not
as-traded); a name turning ineligible is sold at its last price. Unresolved
details take labeled defaults that are counted in the dossier; only missing
inputs or an unmappable signal block a run.

## Limitations stated in every dossier

Survivor universe (current members with history; 19 names without Yahoo data
and PARA excluded); equal weighting; long-only adaptation; every sample is
after publication; 132 months cannot reach t > 3 for realistic spreads; zero
cash yield; Yahoo adjustment conventions; effort reduction not measured;
interpretation scores are preliminary until the group writes its own
reference specifications.
