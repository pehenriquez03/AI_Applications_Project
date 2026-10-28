# WARREN — your research screening agent

IEOR 4723, AI-Augmented Investment Management, Columbia University. Group 6.

WARREN reads an academic finance paper, works out the trading strategy it
describes, tests that strategy on a decade of S&P 500 prices, prices it as a
fund, and writes a research dossier that ends in one of four verdicts:
**Pursue**, **Investigate Further**, **Deprioritize**, or **Unable to Evaluate**.
It is the first screen an investment team would run before spending analyst
time on a paper.

The division of labour is strict. A language model (Claude Sonnet 5.5) does the
reading: it extracts the strategy with page-cited evidence, checks its own
extraction against the paper, and writes the prose of the dossier. Python does
every number: eligibility, signals, portfolios, the backtest, the statistics,
the Monte Carlo, the fees. Every number the model cites in the dossier is
checked against the computed results before the dossier is written.

---

## Quick start

```bash
cd "AI_Applications_Project_philip"
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"      # once
.venv/bin/pytest -q                                              # 204 tests, no network
.venv/bin/streamlit run research_engine/app/streamlit_app.py     # the app at http://localhost:8501
```

The first time the app opens on a machine with no key it asks for one (see
[API key](#api-key)). The nine papers in the app already have cached model
answers, so they run even without a key.

Command line, one paper end to end:

```bash
.venv/bin/python -m research_engine.cli all papers/pdf/bali_cakici_whitelaw_2011_maxing_out_nber14804.pdf
```

Options: `--llm files_only` (never call the model), `--llm api-refresh` (call
the model even when a cached answer exists), `--model`, `--start` and `--end`
(backtest window), `--select` (which experiment in the paper), `--reference`
(a reference specification for the interpretation score).

---

## The pipeline, step by step

Each run is one paper in, one folder under `runs/` out. The nine steps below
are the nine lines the app shows while a run is in progress; each writes its
own files, and `progress.json` is rewritten after every status change so the
app can follow along.

### 1. Read paper

The PDF is opened with `pypdf`, text is extracted page by page and stored as
`paper_text.json` with page numbers, so every later quotation can be checked
against the page it claims. The paper's hash, page count and a title guess go
to `paper_meta.json`. A scanned PDF with no text layer is refused here with a
message to run OCR first.

### 2. Extract strategy

The model reads the page-tagged text three times, independently, and each time
returns a `StrategySpec`: sixteen material fields (signal in words and in
symbols, universe and filters, required inputs, formation window, skip, holding
period, rebalance frequency, sort rule, breakpoints, favored and disfavored
groups, weighting, expected sign, headline result, economic rationale), each
with a page number and a verbatim quote. It also says which experiment in the
paper it chose, whether the strategy is implementable in principle, and lists
the details the paper leaves unresolved.

The three answers are combined field by field into a consensus spec and an
agreement score, and every quote is located on its cited page (`validation.json`
records exact, loose or missing matches). The app shows all of this in the
Extraction tab. The answer key used for grading is never part of any prompt.

### 3. Validate and adapt

A second model call, the check pass, reads each extraction against the paper
and reports contradictions with quotes. Then Python turns the paper's strategy
into an experiment the engine can run (`experiment_spec.json`):

- the signal is mapped to operations in the strategy registry (for example
  "largest daily return in the previous month" becomes `rolling_max_daily_return`);
- the sort, breakpoints, holding period and favored side are set from the paper;
- details the paper does not state take labeled defaults (monthly rebalance,
  equal weights, 20 names minimum per group), and things the paper does that the
  engine cannot do are adapted (value weights become equal weights, a long-short
  spread becomes long-only in the favored group, a six-month holding period
  becomes six overlapping monthly cohorts).

Every default and adaptation is written to `adaptation_log.json` with a reason,
and the counts appear in the dossier. A run stops here, with the reason, only
when a required input is not in the data (fundamentals, futures, options,
volume-based liquidity) or the signal maps to no operation.

### 4. Check data

The frozen price snapshot is loaded (415 S&P 500 names, daily Yahoo prices
2012 to 2025, SPY, Fama-French daily factors), eligibility is counted year by
year, and the data-quality exclusions and patches are recorded in
`data_check.json`.

### 5. Construct portfolios

At each month end the signal is computed for every eligible name from data
through that day, names are ranked and grouped (deciles, quintiles, terciles or
a top percentage, with an optional inner sort), and the favored group becomes
the target portfolio for the next month. Strategies with a holding period
longer than a month run as overlapping cohorts, each holding its own names and
all netted into one set of targets. `signals.parquet` keeps every signal,
rank, group and weight.

### 6. Backtest

A holdings-level simulation from 2015-01-01 to 2025-12-31 (or the window you
choose): signals use the prior month's last close, trades execute at the next
month's first close, each trade pays 10 basis points per dollar (0 and 25 are
also run), weights drift between rebalances, and a name that becomes ineligible
is sold at its last price. Gross and net books, holdings, trades, daily returns
and the two benchmarks (SPY and the equal-weight universe) go to parquet files.
Each decile or quintile group is also backtested on its own.

### 7. Evaluate

`scorecard_strategy.json`: CAGR, volatility, Sharpe, maximum drawdown, returns
relative to both benchmarks with Newey-West errors, alpha against market, size,
value and momentum factors, the cost grid, two sub-periods, the monotonicity of
the ranked groups (Spearman rho against the paper's expected direction),
turnover, concentration and capacity.
`scorecard_interpretation.json`: the extraction compared with a reference
specification written by the group, so the dossier can say how reliably the
paper was read, not only how the strategy performed.

### 8. Monte Carlo

A stationary block bootstrap (1,000 five-year paths, 126-day mean block, the
same sampled dates for the strategy and both benchmarks) feeds a fund model:
$100M, 2% management and 20% performance fee over a 5% hard hurdle with a
high-water mark, five years, discounted at 10%. The output is the probability
that the strategy's incremental fee value is positive against an equal-weight
fund, the quartiles of NPV and ending wealth, and the research cost (measured
model tokens plus 80 hours of development and review and 10 hours a year of
maintenance at $150).

### 9. Dossier

A fixed rubric turns the scorecards into the verdict and its reasons. The
model then writes the narrative (summary and six sections) from `results.json`
only; every number in its text is matched against the results, and any number
the results do not contain is listed as unverified in the dossier. The HTML dossier
(`dossier.html`) holds the verdict, the narrative, the extraction with its
evidence, the adaptation log, all tables, the economics, and the limitations.
`runs/index.html` compares all runs, including the correlation of their excess
returns, so five variants of past-return information are not mistaken for five
independent findings.

---

## The papers

Nine papers are built into the app. Six are the core set, one is the mock
paper the original project was built around, and two are decoys that a correct
reading must report as not runnable on price data.

| Tile | Paper | What the strategy becomes here | Verdict (2015–2025) |
|---|---|---|---|
| MAX | Bali, Cakici & Whitelaw 2011, *Maxing Out* | long the decile with the lowest maximum daily return over the prior month, monthly | Deprioritize: net CAGR 6.3% vs 12.7% equal-weight universe |
| MOM | Jegadeesh & Titman 2001, momentum | long the top decile of six-month past return, held six months as overlapping cohorts | Investigate Further: 14.5% vs 12.7%, P(NPV>0) 76% |
| SEAS | Heston & Sadka 2008, seasonality | long the top decile of the average same-calendar-month return at lags 24 to 60 months | Deprioritize: 8.0% |
| IVOL | Ang, Hodrick, Xing & Zhang 2006, idiosyncratic volatility | long the quintile with the lowest residual volatility against the Fama-French factors | Deprioritize: 11.9% |
| 52WH | George & Hwang 2004, 52-week high | long the top 30% by price relative to the 52-week high, six-month cohorts | Deprioritize: 10.6% |
| FROG | Da, Gurun & Warachka 2014, frog in the pan | top past-return quintile, then the most continuous information tercile, six-month cohorts | Deprioritize: 11.9% |
| OIL | Oil factor, the group's mock paper | long the decile with the highest trailing oil beta; the paper's oracle timing signal is dropped | Investigate Further: 15.8%, flagged for a disputed extraction |
| GPROF | Novy-Marx 2013, gross profitability (decoy) | needs revenues, cost of goods sold and total assets | Unable to Evaluate |
| TSMOM | Moskowitz, Ooi & Pedersen 2012, time-series momentum (decoy) | needs futures across asset classes | Unable to Evaluate |

The PDFs live in `papers/pdf/` (not redistributed; see `papers/README.md`),
the grading key in `papers/answer_key.json`, and the group's reference
specifications in `papers/reference_specs/`. The current reference
specifications are marked preliminary; the interpretation score reads
"preliminary" until the group replaces them.

---

## The app

The page is a terminal layout. Left: the run. Right, top to bottom: the Run
button, the nine paper tiles, the PDF drop zone, the parameters, and the list
of previous runs.

**Running one of the nine papers.** Click a tile (hover shows the paper's
title, authors, page count and first-page text), then press Run. One paper is
selected at a time; clicking the tile again clears it. The nine steps fill in
live. When the run finishes the page reorders: the Summary with the verdict and
the model-written paragraph comes first, then the headline numbers (net CAGR
against both benchmarks, Sharpe, drawdown, alpha t), the cumulative-return
chart, the tabs with the Dossier open first, and the pipeline steps last.

**Previous runs** are one line each, newest first, with the verdict in color.
Click one to open it. A run that was still in progress when the page was
reloaded is picked up where it is.

**Parameters.** Two are yours to set per run: the model (Sonnet 5.5 by
default; Opus 5.5 and Haiku 5.5 call the API because only Sonnet has cached
answers) and the backtest window (any range of months from 2013-01 to 2025-12,
at least 24 months; signals warm up on all data from 2012). The chosen window
is shown in orange in the header and on every run. Everything else is listed
read-only in the panel and in the table below. "Experiment" is optional: leave
it blank and the extraction picks the paper's main experiment and says which.

**Compare runs** (Session panel) opens `runs/index.html`.

---

## Dropping a PDF

Drop any paper and press Run. The paper is read and its strategy extracted
with page evidence exactly as for the built-in papers, step 3 reports whether
the engine could run it (and if not, why), and step 9 writes a model summary
of the strategy, its data needs, and what it would take to run it here.
Steps 4 to 8 are skipped: **no backtest runs on a dropped PDF**. The engine
only does long-only monthly sorts on daily US equity prices, so a strategy that
needs fundamentals, futures, options, intraday or volume data, or another
construction, would be reported as not implementable anyway, and a backtest of
a mis-mapped strategy would be worse than none. The run shows an "Extraction
only" chip, a single Extraction tab, and the summary.

The extraction and the summary go through the API, so a dropped PDF needs a
key. The uploaded file is kept under `runs/_uploads/` with its original name.

---

## API key

The model steps (extraction, check pass, summary, narrative) need an Anthropic
key. **The default is the key shared in class.**

On a machine with no key the app opens on a key setup step:

1. choose "Class key (default)" or "Personal key";
2. paste the key (they start with `sk-ant-`);
3. "Test and save" makes one five-token call; a rejected key, a key without
   model access, or a network failure is reported and nothing is stored;
4. on success the key is written to `.env` in the repository root (gitignored,
   readable by the owner only) with a label, and the header shows
   "class key …" or "personal key …" with the last four characters only.
   "Keep for this session only" writes nothing to disk.

"Continue without a key" skips the step; the cached runs still open. The
Session panel replaces or removes the key later. `.env.example` shows the two
lines for anyone who prefers to write the file by hand. Each run records the
label (not the key) in `run_config.json`.

Costs: one full run of a paper is about 300,000 input tokens and 11,000 output
tokens, under a dollar at Sonnet prices; the tokens are counted into the
research cost in the economics. Every model answer is cached under
`cache/llm/` keyed by model, prompt version, schema and the exact prompt, so
repeating a run costs nothing. Never paste the key into a notebook, slide,
chat message or commit.

---

## Parameters and conventions

| Parameter | Value | Where |
|---|---|---|
| Universe | 415 current S&P 500 members with history from 2012 (survivor universe; 19 names without Yahoo data and PARA excluded; one Danaher day patched) | `data/snapshot_2026-10-28/manifest.json`, `contracts.py` |
| Backtest window | 2015-01-01 to 2025-12-31 by default; any 24+ month range from 2013-01 in the app | `contracts.py`, app |
| Warm-up | all snapshot data from 2012-01-01 | `WARMUP_START` |
| Signal and execution | signal through the prior month's last close; execute at the next month's first close | `ExperimentSpec.execution` |
| Weighting, rebalance | equal weights, monthly | fixed |
| Holding | one month, or overlapping monthly cohorts when the paper holds longer | `holding_months` |
| Trading cost | 10 bps per dollar traded; 0 and 25 reported | `COST_BPS_BASE`, `COST_BPS_GRID` |
| Group floors | 20 names per group, 15 in a double sort; 252 days of history to be eligible | `MIN_NAMES`, `MIN_NAMES_DOUBLE_SORT`, `MIN_HISTORY_DAYS` |
| Price floors | none (snapshot prices are split-adjusted, not as traded) | `validation/adapt.py` |
| Benchmarks | SPY and the equal-weight universe; economics against the equal-weight fund | `BENCHMARKS`, `ECON_BENCHMARK` |
| Fund | $100M, 2/20, 5% hard hurdle, high-water mark, annual crystallization, 5 years, 10% discount | `FUND` |
| Monte Carlo | 1,000 paths, 126-day mean block, sensitivity at 63 and 252, fixed seed | `MONTE_CARLO` |
| Research cost | measured tokens + 80 h development and review + 10 h/year maintenance at $150 | `RESEARCH_COST` |
| Model | `claude-sonnet-5-5`, three extractions and three check passes per paper, one narrative | `MODEL`, `pipeline.py` |

---

## What a run produces

```
runs/<sha12>_<timestamp>/
  run_config.json          model, window, extraction-only flag, key label
  paper_meta.json, paper_text.json
  spec_raw_1..3.json       the three extractions      spec_consensus.json   the merged one + agreement
  check_pass_1..3.json     the model's check of each  validation.json       evidence check + consistency
  experiment_spec.json     what was tested            adaptation_log.json   every default and adaptation
  data_check.json, plan.json, signals.parquet, holdings.parquet, trades.parquet
  returns.parquet          daily: gross, net, SPY, EW universe, each ranked group
  scorecard_strategy.json, scorecard_interpretation.json
  montecarlo.parquet, economics.json
  results.json             everything the dossier may cite
  narrative.json           the model's text, with its verification record
  dossier.html             the report
  strategy_summary.json    extraction-only runs instead of the narrative
  llm_usage.json           every model call: tokens, cached or not, source (api / session)
  progress.json, run.log
```

The nine demo runs are committed under `runs/demo_*`. Their model answers were
produced during the build night by Claude sessions executing the exact prompts
(source "session" in `llm_usage.json`); MAX and momentum have since been rerun
through the API with identical backtest results.

---

## How it was verified

204 tests run without network access: schema and contract tests, engine
invariants (cash accounting, cost arithmetic, drift, forced sales), the
cohorts, the operations, the fee engine against the original project's totals,
the app through Streamlit's test harness, and an independent reimplementation
of the MAX strategy in thirty lines of pandas that matches the engine's returns
to 1.5e-15. The two decoys come back "Unable to Evaluate" with the right
missing inputs. Everything decided during the build is in `BUILD_LOG.md`; the
plan the group agreed is `BUILD_PLAN_FINAL.md`.

---

## Limitations stated in every dossier

Survivor universe (today's members with history, so delisted names are
missing); equal weighting; long-only adaptations of long-short papers; every
test is after publication, so a weaker result than the paper's is expected;
132 months cannot reach t-statistics above 3 for realistic spreads; zero cash
yield; Yahoo adjustment conventions; interpretation scores are preliminary
until the group writes its own reference specifications; the effort an analyst
would save is asserted, not measured.

---

## Repository map

```
research_engine/
  contracts.py      schemas (StrategySpec, ExperimentSpec, Progress), constants, file names
  pipeline.py       the nine stages, run_config, extraction-only runs; writes progress.json
  cli.py            all | resume | index | export-prompts | app
  documents/        PDF -> page-tagged text
  extraction/       prompts, Claude client with cache, evidence check, consensus, strategy summary
  validation/       consistency checks, defaults, adaptation policies (paper -> experiment)
  data/             snapshot builder, loaders (eligibility, patches, exclusions)
  strategies/       operation registry (ops/), construction, cohorts
  backtest/         holdings-based engine, benchmarks
  evaluation/       strategy scorecard, interpretation scorecard, statistics
  economics/        fee engine (from the original project), bootstrap, fund economics
  reporting/        results assembly, rubric, narrative with number check, HTML dossier and index
  app/              the Streamlit app, key intake, helpers, paper registry
tests/              204 tests
data/snapshot_*/    frozen Yahoo + Ken French snapshot (gitignored; manifest and quality report inside)
papers/             PDFs (gitignored), answer key (grading only), reference specifications
runs/               one folder per run; demo_* committed; index.html compares runs
cache/llm/          every model answer, keyed by model, prompt and schema
legacy/             the original project (the optimizer, notebook and data) unchanged
BUILD_PLAN_FINAL.md the agreed plan      BUILD_LOG.md  what happened and every decision taken
```
