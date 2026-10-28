# Project plan handoff: research paper to strategy engine

Written 2026-10-28 by Claude (Fable 5.1) with Philip Henriquez, for review by
Astra. This supersedes `../AI_Applications_Project/HANDOFF_ASTRA.md` and
`../AI_Applications_Project/ASTRA_PROMPT.md` as the scope authority. Those two
files remain useful background on the original codebase.

Course: IEOR 4723, AI-Augmented Investment Management, Columbia. Group of
four. Roughly three to four weeks of build time remain.

Rules for the reviewer: review and planning only. Do not implement, download
data, commit, push, deploy, or make live model calls. Never put
`papers/answer_key.json` or any reference specification into a model prompt.
Do not redistribute the research PDFs.

---

## 1. Objective

Build an engine that takes one academic finance paper per run and:

1. Uses an LLM to extract the trading strategy the paper describes into a
   structured specification with page-cited evidence.
2. Has a human verify, label, and approve that specification.
3. Executes the approved specification with reviewed Python code (the LLM
   never writes or runs code).
4. Backtests the strategy with holdings-level accounting on fourteen years of
   US large-cap data.
5. Reports two separate verdicts:
   - **Strategy verdict:** do performance, risk, and fund economics justify
     further investigation?
   - **Interpretation verdict:** did the LLM translate the paper accurately,
     and where did a human have to step in?

A fee model converts the backtest into fund economics so the strategy verdict
is expressed in dollars and break-even assets under management, not only in
Sharpe ratios.

This is not an extraction benchmark and not a data-feasibility classifier.
Those are components. The deliverable is the end-to-end engine plus the
evaluation of nine papers run through it.

### Course fit

The professor's example sheet (`../group project-261009.pdf`) shares one
template across its seven examples: an LLM turns a document into executable
logic, code runs it, results are verified against ground truth, and the
write-up states where the model was right and where a human had to intervene.
Example 1 (methodology document to screening logic) is the structural match;
example 3 (LLM features versus a baseline, with the backtesting discipline)
supplies the evaluation standard. The session three factor checklist is the
bar for the strategy evaluation: prior economic story written before the
backtest, monotonic quantile returns, t-statistic above 3 per Harvey-Liu-Zhu,
survives transaction costs, persists out of sample, not subsumed in a spanning
regression, sensible capacity.

---

## 2. Decisions already made (do not reopen without a reason)

Settled with Philip on 2026-10-28.

| # | Decision | Choice |
|---|---|---|
| 1 | Universe | Keep the 434-name survivor universe. Include each stock only from its S&P 500 add date (`Hist/constituents.csv`, "Date added"). Label the survivorship bias in every report. Add one sensitivity table with and without the 151 post-2010 additions. |
| 2 | Weighting | Equal weighting everywhere, labeled as an adaptation. The LLM is scored on flagging when a paper's headline result is value-weighted. |
| 3 | Paper set | Six cores, two decoys, one legacy demo (section 4). |
| 4 | LLM | Claude via the Anthropic Python SDK, structured outputs validated against a Pydantic schema, behind a thin provider interface. Every response cached to disk so grading and the demo need no API key. |
| 5 | Reference specifications | One member drafts each paper's reference spec from the PDF before seeing any LLM output; a second member checks it against the paper; the preliminary answer key is the tiebreaker. **No timed human-effort baseline.** The write-up states that effort reduction was not measured. |
| 6 | Economics | $100M AUM, 2% management and 20% performance fee, 5% hard hurdle, high-water mark, five-year horizon, 10% discount rate. Headline is break-even AUM and probability that incremental fee income covers development cost. |

Consequence of decision 5: the proposal's promise to "compare research
efficiency and costs against manually constructed benchmarks" is withdrawn.
The interpretation verdict covers accuracy, evidence validity, and required
corrections only.

---

## 3. Verified state of the code and data

Checked by Claude on 2026-10-28 against the development fork
`AI_Applications_Project_philip/` (identical to the original checkout at
commit 579cee5; `lib/` and the notebook diff clean).

**Data.**
- `Hist/`: 434 files named `<TICKER> Hist.csv`, each 3,588 rows from
  2010-01-04 to 2024-04-05, one column `Close <TICKER>`. No missing values.
  The row count matches the NYSE calendar.
- The column is a **dividend- and split-adjusted close** (AAPL opens at 6.47
  in January 2010 against a split-adjusted actual close near 7.64; AT&T at
  7.45 against roughly 28). Raw prices and highs are **not** in the repo.
- `Hist/UniList.xlsx` lists 406 names; 28 of the 434 histories have no
  exposure row in `SampleData/Master.xlsx`.
- `Hist/constituents.csv` has 503 current members with add dates. 69 current
  members have no history file (post-2010 listings). Of the 434 with history,
  151 joined the index after 2010-01-01 and 83 after 2017-01-01. No stock
  that was ever removed from the index is present. This is a survivor
  universe in the strict sense.
- `SampleData/Master.xlsx`: static factor exposures (MKTB, CCDB, OILP, ELEC,
  CONS, EMPR) for 406 names, estimation window undocumented. Used only by the
  legacy oil demo.
- `Factors/*.csv`: monthly macro series. `SampleData/SPY.csv`: daily SPY.
- No market capitalization, no shares outstanding, no volume, no risk-free
  series, no Fama-French factors.

**Code (`lib/`, 2,247 lines).**
- `fees.py`: 2/20 fee engine with hurdle types, high-water mark,
  crystallization frequency. Correct. **Keep.**
- `metrics.py`: annualized return, volatility, Sharpe, drawdown. **Keep.**
- `optimizer_oos.py`: SLSQP exposure-target optimizer. Keep for the legacy
  oil demo only.
- `walkforward_oos.py`: rebalance dates, regime-return precompute, backtest
  stitching. The portfolio-return helper `_portfolio_return_for_period`
  computes `period.mul(weights).sum()`: daily returns times fixed weights,
  which is a daily-rebalanced constant-mix portfolio with no holdings, cash,
  drift, trades, or costs. Line 383 calls `pd.unique` on a Python list and
  raises `TypeError` on pandas 3. **Legacy only.**
- `signals.py`, `data.py`: synthetic R² signal and forward oil target.
  Legacy only.
- `smart_beta_llm_wrapper.py`: OpenAI Responses API call returning a
  `FactorInterpretation`; the only field used downstream is a factor code
  that must equal `OILP`. **Replace.**
- `backtest.py`, `experiment.py`: dead code referencing a `Portfolios/`
  directory that does not exist. **Delete.**
- `Strategy_Economics_main_integrated.ipynb`: 59 cells. Cell 5 uses Windows
  backslash paths. Cell 50 sets `starting_aum=100_000` with a comment saying
  $100mm. Cell 55's `fast_xirr` returns NaN for runs that never recover the
  signal cost and the median drops them silently. Cell 23 raises unless the
  factor is `OILP`.
- `Research/Oil_Factor_Mock_Research_Paper.pdf` contains a section titled
  "Explicit instructions for an LLM factor-extraction test" with the expected
  answer. Its extraction result is a smoke test, not evidence.
- No `requirements.txt`, no `.gitignore`, `__pycache__` committed, an
  `OPENAI_API_KEY` is required to run the notebook at all.

**Papers (`../AI_Applications_Project/papers/`).** 13 PDFs in `pdf/`,
checksums in `MANIFEST.sha256`, 15-entry preliminary answer key in
`answer_key.json` (field values from paper text, each headline tagged
`high` or `verify`). The key is **not** the reference specification; see
decision 5.

---

## 4. Paper set

| Role | Paper | Method | Engine machinery it exercises | File | Status |
|---|---|---|---|---|---|
| core | Bali, Cakici, Whitelaw 2011 | MAX: max daily return in prior month, deciles, 1-month hold | one-month formation and hold, decile sort | `bali_cakici_whitelaw_2011_maxing_out_nber14804.pdf` | in collection; EW variant reported in Table I Panel B; excluding sub-$5 stocks strengthens the EW spread |
| core | Jegadeesh, Titman 2001 | 6-month return, deciles, 6-month hold, overlapping cohorts | cohorts, skip month, $5 and size filters | `jegadeesh_titman_2001_momentum_nber7159.pdf` | in collection |
| core | Heston, Sadka 2008 | same-calendar-month return 12 months ago, deciles, 1-month hold | lagged values; 14 years allows lags to year 12 | `heston_sadka_2008_seasonality_wp.pdf` | in collection; which lags to use is a review decision |
| core | Blitz, van Vliet 2007 | 3-year weekly volatility, deciles, long low vol | weekly resampling, long window | not downloaded (SSRN login) | drop-in: Ang, Hodrick, Xing, Zhang 2006 idiosyncratic volatility, in collection; note that paper's effect is weak equal-weighted |
| core | George, Hwang 2004 | price / 52-week high, top and bottom 30%, 6-month hold, EW | raw price highs, 30/40/30 breakpoints | `george_hwang_2004_52week_high_jf.pdf` | in collection; **needs raw prices from the data refresh** |
| core | Da, Gurun, Warachka 2014 | 12-2 momentum double-sorted on information discreteness, 6-month hold | conditional double sort, cohorts | `da_gurun_warachka_2014_frog_in_the_pan_rfs.pdf` | in collection |
| decoy | Novy-Marx 2013 | gross profitability | must return cannot-implement: needs fundamentals | `decoy_novy_marx_2013_gross_profitability_nber15940.pdf` | in collection |
| decoy | Moskowitz, Ooi, Pedersen 2012 | time-series momentum on 58 futures | must return cannot-implement: wrong asset class | `decoy_moskowitz_ooi_pedersen_2012_time_series_momentum_jfe.pdf` | in collection |
| legacy | mock oil paper | maps to OILP exposure | existing optimizer path, unchanged | `Research/Oil_Factor_Mock_Research_Paper.pdf` | smoke test only |
| stretch | Han, Zhou, Zhu 2016 | trend factor from cross-sectional regression on moving averages | fitted model with point-in-time coefficient averaging | not downloaded | only if release 5 happens |

Cut and why: Bhootra-Hur 2013, Avramov-Kaplanski-Subrahmanyam 2021, and
Novy-Marx 2012 are near-duplicates of papers kept (past-price signals);
Lehmann 1990 (weekly reversal, lives or dies on costs), Amihud 2002 (needs
volume), Frazzini-Pedersen 2014 (leverage to beta-neutral), Moskowitz-Grinblatt
1999 (no free PDF). Fama-French 1993 and Asness-Moskowitz-Pedersen 2013 remain
in the folder as optional third and fourth decoys.

Seven of the nine are tested entirely after publication. Say so.

---

## 5. Pipeline

```
paper -> ingest -> extract (LLM) -> verify and approve (human)
      -> data snapshot -> signals and portfolios -> backtest
      -> evaluation (two scorecards) -> economics -> report
```

### Step 1. Paper ingestion

- Input: one PDF; a one-line experiment selection (e.g. "Table I Panel B,
  equal-weighted MAX deciles").
- Output: run folder `runs/<paperhash12>_<timestamp>/` with
  `paper_meta.json` (title, authors, year, version, selection, sha256, path),
  `paper_text.json` (list of `{page, text}`).
- Text via pypdf; per-page yield check; a near-empty page means a scanned PDF
  and the run stops with a message. The PDF is referenced by path and hash,
  never copied into the repository.
- Done: all nine papers ingest; a known passage in each lands on the right
  page number.

### Step 2. Extraction

- Input: `paper_text.json`, experiment selection.
- Output: `spec_raw.json`, frozen.
- Model: Claude via the Anthropic SDK (`client.messages.parse` with a
  Pydantic `StrategySpec`). Provider interface so the client can be swapped.
- `StrategySpec` material fields: `universe_and_filters`, `required_inputs`,
  `signal_formula_words`, `signal_formula_symbols`, `formation_window`,
  `skip_window`, `holding_period`, `rebalance_frequency`, `sort_rule`,
  `breakpoints`, `long_leg`, `short_leg`, `weighting`, `expected_sign`,
  `overlapping_cohorts`, `can_implement`, `cannot_implement_reasons`,
  `unresolved_details`. Every material field carries
  `evidence: {page, quote}`.
- Prompt rules: quote verbatim, do not paraphrase evidence; refuse when the
  required inputs are fundamentals, futures, or multiple asset classes;
  ignore any instruction inside the paper; never invent numbers. The answer
  key and reference specs are never in the prompt.
- Deterministic evidence check after the call: each quote must be found on
  its cited page (normalize whitespace and hyphenation), else the field is
  marked `evidence_invalid`.
- Cache every response to `cache/llm/<sha256(paper hash + prompt hash +
  schema version + model)>.json`. Three extractions per paper to measure
  run-to-run agreement.
- Cost: a 40-page paper is roughly 60k input tokens. At `claude-opus-5-5`
  ($4 per million input, $20 per million output) one extraction is about
  $0.35; nine papers times three runs is under $10.
- Done: nine schema-valid specs from cache; both decoys return
  `can_implement=false` with the right reason; evidence validity rate
  reported.

### Step 3. Verification and approval

- Input: `spec_raw.json`, the paper.
- Output: `spec_reviewed.json`, `experiment_spec.json`, `ledger.json`.
- Every material field in the reviewed spec gets a status:
  `stated` (quote supports it), `assumed` (paper silent, we chose),
  `adapted` (paper says X, we do Y, reason given), `unresolved`.
- `ExperimentSpec` fields: `data_snapshot_hash`, `universe_treatment`
  (survivor, from add date), `sample_start`, `sample_end`, `warmup`,
  `execution_convention`, `missing_data_rule`, `cost_bps`, `benchmark`,
  `risk_free_series`, `weighting_adaptation`, `economic_parameters`.
- Gate: a spec with any `unresolved` material field cannot run.
- Reviewer tool: command-line or a minimal page that shows each field beside
  its quote; the reviewer edits the value and sets the status. A second
  member checks before approval.
- `ledger.json` is the diff between raw and reviewed with statuses and
  reasons. This is the human-judgment ledger the course examples emphasize.
- Done: both release-1 and release-2 papers have approved specs and a ledger
  listing every change with a reason.

### Step 4. Data

- One script pulls a frozen Yahoo snapshot via yfinance for the 434 names
  plus the 69 missing current members: raw open, high, low, close, adjusted
  close, volume, dividends, splits, full available history. Plus Ken French
  daily factors (Mkt-RF, SMB, HML, MOM) and the risk-free rate.
- Storage: Parquet under `data/snapshot_<date>/` with `manifest.json`
  (download date, ticker list, per-file sha256, survivorship label).
- Validation: no non-positive prices; raw and adjusted agree on split dates;
  calendar matches NYSE days; tickers with gaps above a threshold flagged.
- Derived panels: daily returns from adjusted close; raw close and high
  panel for price-level signals; membership mask from add dates; calendar.
- Rule: no refetch after the manifest is written. Yahoo re-adjusts the
  adjusted close on every dividend, so a refetch changes results.
- Yahoo cannot supply delisted names or pre-2019 shares outstanding. That is
  why decisions 1 and 2 stand.
- Done: snapshot validates, manifest exists, loaders tested against three
  hand-checked tickers.

### Step 5. Signals and portfolios

- A registry of reviewed operations, each a pure function of data strictly
  before the information date.
  - Time series: `cum_return(window, skip)`, `rolling_max_daily_return(window)`,
    `rolling_std(window, freq)`, `lagged_value(lag_months)`,
    `price_to_rolling_high(window)`, `residual_vol(window, market)`.
  - Cross section: `quantile_sort(n, breakpoints)`, `double_sort(outer, inner)`,
    `rank`.
  - Portfolio: `equal_weight(group)`, `long_minus_short`,
    `cohorts(hold_months)` with one-kth of capital per cohort and the
    aggregate as their sum.
- The approved spec compiles into a plan naming operations and parameters.
  The LLM never produces code.
- Conventions in the ExperimentSpec: signals use data through the last
  trading day of month t-1; positions set at the first trading day of month
  t; a leg must hold at least 20 names or the date is skipped and logged.
- A paper needing an operation not in the registry (e.g. the trend factor's
  cross-sectional regression) gets a reviewed module with its own tests
  before it is registered.
- Done: MAX and momentum signals match hand calculations for three stocks on
  two dates; the cohort aggregate equals the mean of live cohorts to machine
  precision.

### Step 6. Backtest

- Input: target weights per rebalance date.
- The engine holds shares. At each rebalance: target shares from weights and
  NAV at execution prices; trades are the difference; cost in basis points
  of traded notional; then daily mark to market until the next rebalance, so
  weights drift.
- Long-short: self-financing, 100% long and 100% short gross; financing and
  borrow cost set to zero and labeled as such.
- Output: daily holdings, cash, trades, turnover, gross and net returns,
  realized weights before each rebalance.
- Invariants (tests): NAV equals holdings plus cash daily; no trades on
  non-rebalance days; a zero-cost daily-rebalanced run reproduces the
  constant-weight approximation of the old helper; costs reduce net return
  monotonically; **changing any price after a rebalance date leaves that
  date's holdings byte-identical.**
- Done: invariants pass; the MAX run reconciles to a hand-built spreadsheet
  for one quarter.

### Step 7. Evaluation

Strategy scorecard per paper:
- mean monthly return and t-statistic for each decile and the spread,
  Newey-West lags equal to the holding period in months;
- Spearman correlation of decile rank with decile mean return (monotonicity);
- annualized return, volatility, Sharpe, maximum drawdown, turnover;
- net results at 0, 10, and 25 basis points;
- realized market beta; spanning regression on Mkt-RF, SMB, HML, MOM;
- sub-period split at 2017-01-01;
- sample label: post-publication for every paper except the stretch paper.

Interpretation scorecard per paper:
- material-field agreement with the reference spec;
- evidence validity rate;
- critical errors (a field that changes the strategy) and minor errors;
- corrections the reviewer made (from the ledger);
- run-to-run agreement across three extractions;
- decoys: refusal correct, reason correct.

Returns are never used to grade interpretation.

Expectation to state up front: with roughly 160 monthly observations, a
spread of 0.5% per month at 4% monthly volatility gives a t-statistic near
1.6. Most strategies will not clear the t > 3 bar on this sample. The
deliverable is a defensible verdict, not a confirmation of the anomaly.

### Step 8. Economics

- Input: net daily returns of the strategy and of the benchmark
  (equal-weighted universe).
- Fund: $100M, five years, 2/20, 5% hard hurdle, high-water mark, annual
  crystallization, using `lib/fees.py` with the AUM constant corrected.
- Development cost: measured token spend plus assumed review hours at a
  stated hourly rate; data and compute at zero, stated.
- Uncertainty: stationary block bootstrap, six-month blocks, strategy and
  benchmark resampled jointly, 1,000 paths, fees recomputed per path, common
  random numbers across strategies.
- Outputs: distribution of five-year fee income; incremental fee income over
  the benchmark fund; probability that fee income covers development cost;
  break-even AUM; count of paths with undefined IRR.
- The legacy R² Monte Carlo on the oil demo stays in its own section labeled
  "hypothetical forecast skill" and is not evidence about any paper.
- Done: the oil demo reproduces the old fee numbers with the corrected AUM;
  the MAX economics table renders with uncertainty bands.

### Step 9. Report and reproducibility

- Run folder contents: paper hash, `spec_raw.json`, `spec_reviewed.json`,
  `ledger.json`, `experiment_spec.json`, data manifest hash, code commit,
  signals, holdings, trades, both scorecards, economics table, and a
  limitations block generated from the status labels and the data manifest.
- Static HTML: one page per run, one cross-paper page with the two
  scorecards and the economics table.
- CLI: `ingest`, `extract`, `review`, `run`. A thin Streamlit page is
  optional and last.
- Pinned dependencies, `.gitignore`, no `__pycache__`, a test suite for the
  step 5 and 6 invariants, cached LLM responses committed so the demo runs
  with no key, one-command smoke test on the oil paper.
- Done: a fresh clone runs the smoke test and renders the MAX report with no
  network access.

---

## 6. Proposed repository layout

```
research_engine/
  documents/     ingest.py, pdf_text.py
  extraction/    schema.py (StrategySpec, ExperimentSpec), prompts/, client.py, cache.py, evidence_check.py
  review/        cli.py, ledger.py
  data/          snapshot.py, validate.py, loaders.py
  strategies/    registry.py, ops_timeseries.py, ops_crosssection.py, portfolio.py, compile.py
  backtest/      engine.py, costs.py, invariants.py
  evaluation/    strategy_scorecard.py, interpretation_scorecard.py, stats.py
  economics/     fees.py (moved from lib), bootstrap.py, breakeven.py
  reporting/     render.py, templates/
  legacy/        optimizer_oos.py, walkforward_oos.py, signals.py, data.py, oil demo notebook
tests/
runs/            one folder per run (gitignored except cached demo)
cache/llm/       committed for the demo papers
data/            snapshot (gitignored), manifest committed
papers/          PDFs (gitignored), answer_key.json, reference_specs/
```

---

## 7. Releases and acceptance criteria

| Release | Scope | Done when |
|---|---|---|
| 1 | MAX equal-weighted, end to end, no LLM | hand-computed MAX for three stocks on one month matches; decile weights sum to one; turnover reported; look-ahead test passes; runs at 0 and 25 bps; report shows deciles, spread with NW t-stat, drawdown, turnover, realized beta |
| 2 | Jegadeesh-Titman 6/6 with overlapping cohorts and skip month | cohort invariants pass; cash conserved per cohort; step 1 criteria repeat |
| 3 | Extraction, verification, interpretation scorecard on releases 1 and 2 plus both decoys | cached specs reproduce with no key; decoys refused with correct reasons; ledger rendered; scorecard rendered |
| 4 | Economics and report pages | oil demo reproduces old fee numbers with corrected AUM; MAX economics with bootstrap bands; cross-paper page renders |
| 5 | Expansion: seasonality, low volatility, 52-week high (needs raw prices), frog in the pan; trend factor if time | each paper's spec approved and run; scorecards updated; stop at the deadline and report what is frozen |

Team split for four people: (a) data and backtest accounting, (b) signal
operations and portfolio construction, (c) extraction, verification tool,
and caching, (d) evaluation, economics, and reporting. Everyone writes one or
two reference specifications.

---

## 8. Known limitations to state in the report

- Survivor universe; direction of bias is against the anomalies tested
  (delisted high-MAX, high-volatility, past-loser names are missing, so short
  legs look better than they were).
- Equal weighting only; value-weighted headline results are not replicated.
- Fourteen years of data; most spreads will not reach t > 3.
- Every core paper is tested after publication; no in-sample replication is
  possible with this data.
- Financing and borrow costs on the short leg are zero.
- Effort reduction from LLM assistance is not measured.

---

## 9. Open details, to be decided at spec review, not now

- Decile breakpoints on a 400-name universe (equal-count deciles of ~40
  names is the default).
- A stock leaving the universe mid-holding: sell at last available price,
  log it.
- Blitz-van Vliet weekly volatility: Friday closes or calendar weeks.
- Heston-Sadka: single lag 12 or the average of annual lags available in a
  14-year window.
- Whether to apply a $5 price filter where papers use one (none of the 434
  names is likely to trip it; verify).

---

## 10. What we want from Astra

1. Confirm the objective in section 1 is preserved and that nothing in the
   plan turns the project into a benchmark or a classifier.
2. Challenge the step designs in section 5. Specifically: the execution
   convention in step 5, the long-short accounting in step 6, the Newey-West
   lag rule and spanning regression in step 7, and the joint block bootstrap
   in step 8.
3. Check the release order and acceptance criteria in section 7 for anything
   that cannot be verified as written.
4. Identify any component that should be reused from `lib/` that section 3
   marks as legacy, with a reason.
5. Flag anything that needs more than four people or more than four weeks.
6. Raise only decisions that genuinely require Philip; the six in section 2
   are settled.

Return the review as a document the group can act on.
