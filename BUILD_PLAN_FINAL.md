# Final build plan: research-screening engine

Agreed with Philip on 2026-10-28, incorporating the design agreed with Astra
the same day. This is the plan the build follows. Anything not in here is an
ordinary engineering choice, decided during the build and recorded in the
run's ExperimentSpec.

## Objective

A research-screening engine. One paper per run. An LLM extracts the trading
strategy with page-cited evidence. Python validates it, a second LLM pass
checks it against the paper, and explicit policies adapt it to our supported
long-only experiment, producing a ready or blocked status. No mandatory human
approval: material ambiguity or missing data blocks the run rather than
inventing an assumption. Reviewed Python builds portfolios, backtests,
evaluates, and simulates; the LLM never writes executable code at runtime.
Each run ends in an evidence-backed dossier with one recommendation: pursue,
investigate further, deprioritize, or unable to evaluate. Both the strategy
and the accuracy of the LLM's interpretation are evaluated.

## All decisions

| Area | Decision |
|---|---|
| Universe | The existing 434 names, prices refreshed through 2025 from Yahoo; each eligible from its index add date and only with sufficient history; survivor bias labeled everywhere; sensitivity with and without post-2010 additions. The 69 current members without history stay out. |
| Weighting | Equal weight everywhere, labeled adapted where the paper is value-weighted |
| Strategy portfolio | Long-only: hold the paper's favored ranked group. Every group is computed for the monotonicity table. No short leg. |
| Holding | Per paper (table below). Six-month strategies use overlapping monthly cohorts with capital split across six active cohorts; overlapping names are netted before trades. Fallback if time runs out: one-month hold, labeled adapted. |
| Sample | 2015-01 to 2025-12; warm-up from 2012 (three-year weekly volatility); sub-periods 2015-2019 and 2020-2025 |
| Benchmarks | SPY total return (Yahoo adjusted close) and the equal-weighted eligible universe |
| Costs | 10 bps of traded notional baseline; 0 and 25 bps sensitivity |
| Papers | MAX; JT momentum; Heston-Sadka seasonality; Blitz-van Vliet low volatility (Ang et al. drop-in); George-Hwang 52-week high; Da-Gurun-Warachka frog in the pan. Decoys: Novy-Marx 2013, Moskowitz-Ooi-Pedersen 2012. Legacy oil demo untouched. |
| LLM | `claude-sonnet-5-5` via the Anthropic SDK; structured outputs; disk cache keyed by paper, prompt, schema, model; three extractions per paper; one checking call per extraction; one narrative call per run |
| Key | The key in the workspace `.env`, copied to the fork's gitignored `.env`; professor's key swaps in later |
| Reference specs | Prepared independently by the group (one drafts, one checks, answer key tiebreaker); never in any prompt; scorecard shows "preliminary" until they exist |
| Economics | $100M start, 2/20, 5% hard hurdle, high-water mark, annual crystallization, five years, 10% discount; research cost = measured model usage + 80 development/review hours at $150 + 10 maintenance hours per year at $150, all stated as assumptions |
| Monte Carlo | Stationary block bootstrap, mean block six months, same sampled dates for strategy and both benchmarks, 1,000 five-year paths, fees recomputed per path, common seeds across strategies; block sensitivity at three and twelve months |
| Interface | A local Streamlit app is part of the deliverable: drag-and-drop PDFs, Run, live nine-stage progress with per-stage facts and blocked reasons, recommendation card, inline dossier, previous-runs list, key status. Built as checkpoint 7b after the MAX dossier works. CLI (`ingest`, `extract`, `check`, `review`, `run`, `report`, `all`) underneath; static HTML dossiers and index page remain the artifacts. Experiment selection optional: when blank the extraction identifies the paper's main experiment and shows it. |
| Repo | Branch `engine` in `AI_Applications_Project_philip`, one commit per verified checkpoint, no push |
| Data | One frozen Yahoo snapshot (434 tickers raw and adjusted, SPY, Fama-French daily factors and RF) with a checksum manifest and a data-quality report; no refetch |

## Strategies as implemented

| Paper | Signal | Selection | Holding |
|---|---|---|---|
| MAX | largest daily return in the previous month | lowest decile | 1 month |
| Momentum | cumulative return over the previous six months | highest decile | 6 months, overlapping cohorts |
| Seasonality | return in the same calendar month one year earlier | highest decile | 1 month |
| Low volatility | volatility of ~3 years of weekly returns | lowest decile | 1 month |
| 52-week high | price / preceding 52-week high (raw prices) | highest 30% | 6 months, overlapping cohorts |
| Frog in the pan | past 12-2 return, then information discreteness within winners | sequential quintiles: top past-return quintile, then lowest-ID quintile | 6 months, overlapping cohorts |

Pending verification before each is frozen: frog-in-the-pan formation-window
endpoints; low-volatility return convention (simple vs excess); 52-week-high
price convention (raw close vs raw high, adjusted for splits only).

## Conventions fixed for every paper

1. Signal uses information through the previous month's final close.
2. Groups are equal-count sorts of eligible names at that date; a group must
   hold at least 20 names or the month is skipped and logged.
3. Execution at the next month's first trading-day close.
4. New weights apply only to returns earned after execution; existing
   holdings earn returns until the trade.
5. Returns from Yahoo adjusted close; price-level signals from raw prices.
6. A name that becomes ineligible mid-holding is sold at its last price and
   logged.
7. Fully invested; cash earns zero.
8. Cohort convention: each month a new cohort receives 1/6 of portfolio
   value; cohorts are netted to one target weight vector before trading.
9. Corporate actions: returns from Yahoo adjusted close (dividends and
   splits folded in); price-level signals from raw prices adjusted for
   splits only, rebuilt from the split series; ticker changes as Yahoo
   resolves them today; spin-offs appear only through the adjusted close;
   no stock in the 434 stopped trading. The data-quality report lists every
   ticker where raw and adjusted series disagree on a split date.

## Pipeline and build tasks

### Step 0. Scaffold (1 h)
Branch `engine`. `.gitignore` (`.env`, `data/snapshot_*`, `runs/`,
`__pycache__`). `pyproject.toml` with pinned deps. Package `research_engine/`.
Move `lib/fees.py`, `lib/metrics.py` into the package; move the rest of
`lib/` and the notebook to `legacy/`; delete `lib/backtest.py`,
`lib/experiment.py`, committed `__pycache__`. Copy `.env`. Commit.

### Step 1. Read the paper (1 h)
`documents/ingest.py`: PDF to `paper_text.json` (page, text) and
`paper_meta.json` (sha256, title, selected experiment); run folder
`runs/<hash12>_<ts>/`; scanned-PDF detection. Test on all nine papers.

### Step 2. Extract the strategy (2 h)
`extraction/schema.py`: `StrategySpec` (universe and filters, required
inputs, signal formula in words and symbols, formation window, skip,
holding period, rebalance, sort rule and breakpoints, favored group,
weighting, expected sign, can_implement, cannot_implement_reasons,
unresolved_details), each material field with `evidence: {page, quote}`.
`extraction/client.py`: Sonnet via `messages.parse`, cache, usage file.
`extraction/evidence_check.py`: quote found on cited page. Three runs per
paper, agreement computed. Decoys must refuse.

### Step 3. Validate and adapt (2 h)
`validation/consistency.py`: completeness and consistency checks in Python
(windows positive, favored group consistent with expected sign, required
inputs available in the snapshot). `validation/check_pass.py`: a second
Sonnet call that receives the spec and the page text and flags fields it
believes contradict the paper. `validation/adapt.py`: explicit policies
(equal weight; long-only favored group; cohort or monthly holding; our
universe and dates) applied to produce `experiment_spec.json` and
`adaptation_log.json`. Unresolved or silent fields take the defaults in the
table below, labeled `assumed`; the run continues. Status `blocked` only
when `can_implement` is false after the check pass agrees, a required input
is missing from the snapshot, or the signal maps to no registry operation.
`review/cli.py`: optional human override of any default or adapted value,
with a reason; writes `ledger.json`.

Defaults for unresolved details: no price filter unless stated; one year of
history to be eligible (or the signal window if longer); 15 trading days
minimum in a formation month; no skip; one-month hold; monthly rebalance;
deciles with equal-count breakpoints; 30% for top/bottom-percent sorts;
favored group from the paper's headline table, else all groups run and the
rubric marks "investigate further"; equal weight; simple weekly returns from
Friday closes for volatility; raw split-adjusted daily high for the 52-week
high; momentum formation ends at the prior month's last close; frog-in-the-pan
window t-12 to t-2; ineligible mid-hold sold at last price; cash zero yield.
The implementation-reliability flag turns amber when more than three
material fields are assumed.

### Step 4. Check the data (2-3 h)
`data/snapshot.py`: yfinance for the 434 tickers (raw OHLC, adjusted close,
volume, dividends, splits), SPY, Fama-French daily factors and RF; Parquet
under `data/snapshot_<date>/`; `manifest.json` with sha256 per file and the
survivorship label. `data/validate.py`: positive prices, split agreement
between raw and adjusted, NYSE calendar, gap flags, Yahoo availability per
ticker; writes `data_quality.json`. `data/loaders.py`: returns panel, raw
price panel, membership mask (add date and history requirement), calendar.
Per run: confirm required inputs exist; else block.

### Step 5. Construct portfolios (3-4 h)
`strategies/registry.py`: `rolling_max_daily_return`, `cum_return(months,
skip)`, `lagged_month_return`, `rolling_vol(window, freq)`,
`price_to_rolling_high`, `info_discreteness`, `group_sort`, `double_sort`,
`equal_weight`, `cohorts(hold_months)`. `strategies/compile.py`: experiment
spec to a plan. `strategies/portfolios.py`: dated signals, group
memberships, selections, target weights, cohorts. Hand-check MAX and
momentum for three stocks on two dates.

### Step 6. Run the backtest (3 h)
`backtest/engine.py`: holdings in shares; rebalance at month-start close to
netted target weights; cost in bps of traded notional; daily mark to market;
outputs daily value, holdings, cash, trades, turnover, gross and net returns.
Same engine for every group and both benchmarks. `tests/test_invariants.py`:
value identity; trades only on rebalance days; cost monotonicity; new
weights earn nothing before execution; look-ahead mutation test.

### Step 7. Evaluate (3 h)
`evaluation/strategy.py`: cumulative and annualized return, volatility, max
drawdown and series, Sharpe, relative return to both benchmarks, CAPM beta,
four-factor alpha, turnover and costs, holdings count and concentration;
robustness at 0/10/25 bps, sub-periods, Newey-West t-statistics and
confidence intervals on benchmark-relative monthly returns and alpha,
ranked-group table with Spearman monotonicity, universe sensitivity,
coverage gaps. `evaluation/interpretation.py`: field agreement with the
reference spec (or preliminary key, labeled), evidence validity, critical
and minor errors, check-pass disagreements, run-to-run agreement, decoy
refusal.

### Step 8. Monte Carlo and economics (2 h)
`economics/bootstrap.py`: stationary bootstrap, joint dates, 1,000 paths of
five years, common seeds. `economics/fund.py`: fee engine per path for the
strategy fund and the benchmark fund; investor ending wealth and drawdown
reported separately from manager fee income; incremental fee NPV at 10%
less research cost; fraction of paths positive; break-even AUM only when
expected incremental economics are positive, else "none", with uncertainty
stated near zero. Block sensitivity 3/6/12 months. Fee accounting tests.

### Step 9. Research report (3 h)
`reporting/rubric.py`: flags for implementation reliability, historical
performance, robustness, economics, limitations, applied as declared rules
to produce the recommendation. `reporting/narrative.py`: one Sonnet call
over `results.json`; every number it states must exist in the JSON
(checked); it explains, it does not decide. `reporting/render.py`: HTML
dossier per run (paper's proposal, extraction and evidence, adaptations,
data and assumptions, construction, performance, Monte Carlo, interpretation
accuracy, recommendation with limitations) plus downloads; `runs/index.html`.

## Build order and checkpoints

| # | Steps | Checkpoint before commit |
|---|---|---|
| 1 | 0 | package imports; legacy notebook untouched |
| 2 | 4 | snapshot validates; manifest and data-quality report written; loader tests pass |
| 3 | 5, 6 (monthly path) | MAX hand-check passes; invariant tests pass; MAX net returns 2015-2025 and both benchmarks produced |
| 4 | 7 | MAX strategy scorecard and robustness tables render |
| 5 | 8 | MAX economics with distributions, fee tests, break-even |
| 6 | 1, 2, 3 | nine papers ingested; nine specs from cache; check pass run; decoys blocked; MAX ready |
| 7 | 9 | MAX HTML dossier with narrative; index page |
| 7b | UI | Streamlit app: drop PDF, Run, live stage progress via pipeline callbacks, blocked reasons shown, recommendation card, inline dossier; MAX runs from cache instantly and a fresh paper end to end in about four minutes |
| 8 | 5 cohorts; remaining papers | cohort invariants; then seasonality, low volatility, momentum, 52-week high, frog in the pan, each: extract, validate, run, dossier |

Expected: checkpoints 1-7 in one session; checkpoint 8 as far as time
allows. If cohorts are not finished, the three six-month papers run on the
one-month variant, labeled adapted, and the dossier says so.

## Deviations from Astra's handoff (RESEARCH_ENGINE_BUILD_HANDOFF_2026-10-28.md)

Decided by Philip on 2026-10-28:
- Unresolved fields take labeled defaults and the run continues; blocking is
  reserved for missing inputs or an unmappable signal. The count of assumed
  fields is reported in the rubric and dossier.
- The Streamlit app is in scope as checkpoint 7b, after the MAX dossier.
- Two decoy papers are run to test the blocked path and score refusal.
- Experiment selection is optional; extraction names the main experiment.

## Not built

Short legs, value weighting, delisted names, the trend-factor paper, a
review web page, parameter tuning from the UI, any model-generated code.

## Limitations stated in every dossier

Survivor universe; equal weighting; long-only adaptation; sample entirely
after publication; zero cash yield; Yahoo adjustment conventions; effort
reduction not measured.
