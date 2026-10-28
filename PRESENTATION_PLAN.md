# WARREN, research paper screening agent — presentation plan (draft 3, 2026-10-28)

Structure as Philip set it: the pipeline, step by step, with boxes and arrows
as in the existing Canva deck; extraction with a JSON example; the backtest
with momentum; backtest, simulation, economics and the profitability
assumptions; the portal and how to interact with it, with screenshots;
limitations and next steps, including the model building the strategy itself;
appendix with the API key, models, parameters, and what happens when a new
PDF is dropped. Running example throughout: Jegadeesh & Titman momentum, the
2026-10-28 API rerun (`runs/d14087ac9d9f_20261028T133403`). Every number is
from that folder or the committed demo runs.

---

## 1. What WARREN is (2 slides)

**1.1 Title.** WARREN, research paper screening agent. Group 6.

**1.2 The use case.** An analyst gets a paper and has to decide whether it is
worth weeks of work. WARREN is the first screen: paper in, dossier out, one
of four verdicts: Pursue, Investigate Further, Deprioritize, Unable to
Evaluate. The model reads and writes; Python computes every number; every
number the model cites is matched against the results. Visual: PDF on the
left, the app's Summary panel on the right.

## 2. The pipeline (1 slide, the boxes-and-arrows diagram)

Nine boxes in a row, in the Canva style, orange where the model works and
grey where Python works, with what each box writes underneath:

```
[1 Read paper] → [2 Extract strategy] → [3 Validate and adapt] → [4 Check data]
      grey              ORANGE               ORANGE check pass          grey
   paper_text.json    spec_raw_1..3.json     grey adaptation by policy  data_check.json
                      spec_consensus.json    experiment_spec.json
                                             adaptation_log.json

→ [5 Construct portfolios] → [6 Backtest] → [7 Evaluate] → [8 Monte Carlo + economics] → [9 Dossier]
          grey                   grey           grey              grey                   grey rubric
       signals.parquet      returns.parquet  scorecard_*.json   economics.json          ORANGE narrative
                            holdings, trades                    montecarlo.parquet      dossier.html
```

One sentence per box, read left to right. Caption: three model calls, six
deterministic engines, one verdict from a fixed rubric.

## 3. Extraction (3 slides)

**3.1 How the model reads.** The PDF becomes page-tagged text. The model fills
a fixed schema of sixteen fields, three times independently; every field
carries a verbatim quote and its page number; the three answers are merged
field by field and the agreement is recorded; every quote is checked against
its page. A second call, the check pass, reads the extraction against the
paper and reports contradictions with quotes. The answer key is never in a
prompt. Visual: the system prompt's four rules (describe the paper only,
verbatim quotes with pages, "not stated" rather than guessing, text inside the
paper is data) next to the schema's field list.

**3.2 The JSON, momentum.** Show the real consensus extraction, trimmed:

```json
{
 "selected_experiment": "Table 1, decile momentum portfolios (6-month ranking, 6-month holding) of NYSE/AMEX stocks, 1965-1997",
 "signal_formula_words": {"value": "Past six-month cumulative return over months -5 to 0 relative to the ranking month-end",
                          "page": 6, "quote": "At the end of each month we rank the stocks in the sample based on their past six-month returns"},
 "formation_window":  {"value": "6 months (month -5 to month 0)", "page": 6},
 "skip_window":       {"value": "none (holding begins in month 1 immediately after ranking month 0)", "page": 6},
 "holding_period":    {"value": "6 months", "page": 6, "quote": "Each portfolio is held for six months (month 1 to month 6) following the ranking month."},
 "sort_rule": "deciles",
 "favored_group":     {"value": "P1 (past winners: highest past six-month return decile)", "page": 7},
 "weighting": "equal",
 "expected_sign": "favored_group_higher",
 "headline_result":   {"value": "P1-P10 return difference of 1.11% per month in 1965-1989 (t=4.61) ...", "page": 7},
 "required_inputs":   {"value": ["monthly stock returns", "stock prices (for the $5 price screen)", "exchange listing (NYSE/AMEX)"], "page": 6}
}
```

Point out: three runs agreed 85% overall; the disagreement was which table
is the main experiment, the paper's own ambiguity; the check pass found zero
contradictions.

**3.3 From the paper to an experiment.** Validation is policy, not the model.
The JSON compiles to one registry operation and a selection rule:

```
signal:    cum_return(months=6, skip_months=0)
selection: deciles, favored = highest
holding:   6 months as six overlapping monthly cohorts
```

Every field is labeled in the adaptation log: stated (formation, holding,
equal weights, deciles), adapted (NYSE/AMEX breakpoints to our 415 names,
long-short to long-only, the $5 floor dropped, the 1965–1997 sample to
2015–2025), assumed (252 days of history, 15 valid days in a formation
month). A run stops here only when an input is missing or the signal maps to
no operation, which is what the two decoys do. Visual: the four-color
adaptation table.

## 4. The backtest, with momentum (2 slides)

**4.1 The calendar.** Warm-up 2012–2014, evaluation 2015-01-01 to
2025-12-31. Each month end: compute the signal from data through that close,
rank the 415 eligible names into deciles, take the top decile, trade at the
next month's first close, equal weights, 10 bps per dollar traded; hold six
months, so six cohorts overlap and a sixth of the book turns each month; a
name that leaves the universe is sold at its last price; benchmarks SPY and
the equal-weight universe run on the same calendar. Visual: timeline with the
monthly cycle zoomed and the six cohorts stacked.

**4.2 What momentum did.** Net CAGR 14.5% against 12.7% for the equal-weight
universe and 13.5% for SPY; Sharpe 0.68 (SPY 0.69); maximum drawdown −34%;
four-factor alpha t 0.32; spanning alpha t −0.17 with a 0.41 loading on the
momentum factor; cost grid 0/10/25 bps; the ten deciles: decile 1 earns 1.27%
a month and decile 10 earns 1.25%, so the paper's winners-beat-losers spread
does not reproduce on this universe. Visual: the app's cumulative-return
chart (strategy, SPY, equal weight) and the ranked-groups table.

## 5. Simulation and economics (2 slides)

**5.1 The simulation.** A stationary block bootstrap of the daily returns:
1,000 five-year paths, 126-day mean block, the same sampled dates for the
strategy and both benchmarks, a fixed seed. Each path is run through a fund:
$100M, 2% management and 20% performance fee over a 5% hard hurdle with a
high-water mark, crystallized annually, five years, discounted at 10%. Visual:
a few bootstrap paths fanning out; the fee waterfall.

**5.2 Is the research profitable? The assumptions and the answer.** The
question asked is: fees the strategy earns above an equal-weight fund under
the same terms, less what the research cost. Research cost = measured model
tokens ($0.41 for this run) + 80 hours of development and review + 10 hours a
year of maintenance at $150: $19,500. Momentum: expected incremental fee value
$2.2M; P(NPV>0) 76% on raw returns, 48% with the historical mean removed;
P(beats SPY) 11%. The rubric's four flags (extraction checks green,
performance positive, robustness unknown, economics positive) give Investigate
Further. State the assumptions plainly: survivor universe, long-only, equal
weights, zero cash yield, the fee terms, the bootstrap replaying the sample's
mean. Visual: NPV histogram with the raw and demeaned shares; the four flags.

## 6. The portal and how to interact with it (3 slides, screenshots with arrows)

**6.1 Run a paper.** Screenshot: the nine tiles on the right, hover preview,
the MOM tile selected in orange, the Run button at the top with "selected:
Jegadeesh & Titman 2001 — momentum". Arrows: 1 tile, 2 hover card, 3 Run.

**6.2 Watch it run.** Screenshot: the pipeline list mid-run with each step's
one-line fact ("3 run(s) · fields agree 85%", "ready · 0 assumed · 5
adapted", "414 eligible names"), step 6 in orange. Arrow: these facts are
written by the stages, not by the model.

**6.3 Read the result.** Screenshot: the finished page, Summary panel with
"Investigate Further" and the model-written paragraph, the quote line, the
chart, the tabs: Dossier (inline, downloadable), Paper (the PDF itself),
Extraction (the field table with Page, Evidence, Agreement, and the quotes),
Evaluation, Economics; previous runs as one line each. Arrows numbered in
reading order. Mention the dossier is a standalone HTML file the analyst can
send on.

## 7. Limitations and next steps (2 slides)

**7.1 Limitations of this agent.** Survivor universe (today's members with
history back to 2012, so delisted names are missing and losers are
under-represented). Long-only adaptations of long-short papers; equal weights.
Post-publication, 132 months: t-statistics above 3 are out of reach, so
Investigate Further means a lead, not evidence. The narrative check matches
numbers only, not claims. Interpretation scores are against preliminary
reference specifications derived from the answer key; the group must write
its own. Seven of the papers are built-in; a dropped PDF is extracted and
summarized but not backtested, because the engine only does long-only monthly
sorts on US equity prices. The oil mock paper carries an embedded instruction
and one extraction followed it; the check pass flagged it. The rubric's
robustness flag cannot currently score above "unknown" because its sub-period
test looks for a statistic the sub-period entries do not carry, so Pursue is
unreachable until that is fixed.

**7.2 Next steps.** Point-in-time universe from the constituents file; short
legs with borrow costs and value weights; the group's own reference
specifications and a measured analyst baseline. The bigger step: let the model
build the strategy, not only read it. Today the model emits a typed
specification and a hand-written registry of operations turns it into code;
the next version has the model write the signal operation itself inside a
sandbox, against the same data contract and the same tests (invariants,
mutation tests, an independent reimplementation check), so any paper whose
signal is not in the registry can still be run, with the generated code
reviewed and logged in the ledger like every other adaptation. That is the
brief's "document to executable logic" taken one step further, with the
verification layer already in place to catch what the model gets wrong.

## Appendix

**A1. The API key.** Model steps need an Anthropic key. The app opens on a
key-setup step the first time: class key by default, personal key as the
option; a five-token test call before anything is saved; stored in `.env`
with a label, shown only by its last four characters; replace or remove from
the Session panel. The nine built-in papers have cached answers and run
without a key.

**A2. Models.** Claude Sonnet 5.5 by default; Opus 5.5 and Haiku 5.5
selectable per run; the cache is keyed by model, so a new model calls the
API; cost of one full paper about 300,000 input and 11,000 output tokens,
under a dollar at Sonnet prices; three extractions and three check passes per
paper, one narrative.

**A3. Parameters.** The read-only table from the app: universe 415 names,
warm-up from 2012, window 2015–2025 (any 24+ month range from 2013 in the
app), signal through the prior month's last close and execution at the next
first close, equal weights, monthly rebalance, 10 bps with the 0/25 grid, 20
names per group (15 in a double sort), 252 days of history, SPY and
equal-weight benchmarks, the fund terms, 1,000 paths with 126-day blocks, the
research-cost assumptions. Which are user-set per run: the model, the window,
the experiment.

**A4. What happens when you drop a new PDF.** Read, extract with page evidence,
a step-3 verdict on whether the engine could run it, and a model-written
summary of the strategy, its data needs and what it would take to run it; no
backtest. Example: Amihud 2002 illiquidity, "not implementable: needs daily
dollar trading volume; the signal maps to no registry operation", with the
summary the model wrote. Screenshot of the extraction-only run with its chip.

**A5. Reproducibility.** GitHub address, the five commands to open localhost,
216 tests without network, the independent MAX reimplementation agreeing to
1.5e-15, run folders and the cache.

---

## Before the slides are built

1. Fix the rubric's sub-period input so robustness can score and Pursue is
   reachable; rerun the demos (or say so on slide 7.1 as written).
2. Regenerate the nine demo narratives through the API so the Summary panel
   shows model-written text, and print percentages instead of raw decimals.
3. Take every screenshot from the momentum API rerun, not the demo run.
