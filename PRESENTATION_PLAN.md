# WARREN, research paper screening agent — presentation plan (draft 2, 2026-10-28)

Draft 1 was reviewed by a separate agent briefed with the course brief and the
four session decks. Its main point: the first draft spent eleven of eighteen
slides on mechanism and screenshots and about two on what the brief grades
(document to executable logic, verification against ground truth, a ledger of
human judgment, backtesting hygiene, LLM versus numbers). Draft 2 keeps the
why, how, results order, threads one paper through every stage instead of a
separate walkthrough, and gives the graded criteria their own slides. Main deck
15 slides for 15 minutes; the portal walkthrough moves to the appendix as a
90-second live demo.

Running example: **Jegadeesh & Titman 2001 momentum**, the 2026-10-28 API
rerun (`runs/d14087ac9d9f_20261028T133403`): three extractions, model-written
narrative, every number below from its files. Every number on a slide must be
traceable to a run folder; the reviewer checked these and they hold.

---

## Main deck

**1. Title.** WARREN, research paper screening agent. Group 6. Tagline:
"Python computes every number; the model reads and writes; every number the
model cites is matched to the results." (Not "every sentence checked": the
narrative check matches numeric tokens only.) Visual: the orange badge header.

**2. The analyst's problem.** More papers than a desk can replicate; the
go/no-go is made with the least evidence at the moment it matters most. WARREN
is the first screen: is this paper worth an analyst's weeks on our universe
with our constraints? Four verdicts: Pursue, Investigate Further,
Deprioritize, Unable to Evaluate. Not a trading system, not a replication, not
a judge of academic merit. Visual: the four verdict badges.

**3. The answer for one paper, first.** Momentum: Investigate Further. Net
CAGR 14.5% against 12.7% for the equal-weight universe and 13.5% for SPY;
Sharpe 0.68 (SPY 0.69); max drawdown −34%; alpha t 0.32; P(NPV>0) 76%, 48%
with demeaned returns. Say what the verdict means: a lead worth an analyst's
look, not evidence. Visual: the app's Summary panel and quote line.

**4. The pipeline: who reads, who computes, who decides.** Nine stages,
orange for the model (extract, check pass, narrative), grey for Python (read,
adapt by policy, data, construct, backtest, evaluate, Monte Carlo, rubric).
Caption: "The model reads and writes; Python computes and decides." Course
echo: session 2's hybrid funnel, deterministic code for what can be specified,
model judgment for the rest, humans for what matters; the model never writes
code here, it emits a typed specification that compiles to a reviewed registry
of operations. Visual: flow diagram with the file each stage writes.

**5. Read and extract.** Page-tagged text; a fixed sixteen-field schema; three
independent extractions; verbatim quote and page for every field, located on
its page (100% valid for momentum); consensus with an agreement score (85%
across the three API runs; the three disagreed on which table is the main
experiment, which is the paper's own ambiguity). Course echoes: "the prompt is
the specification"; citations and honesty, the "not stated" rule; session 3's
momentum skip month, which this paper does not use, and the extraction quoted
the page that says so. Visual: three rows of the extraction table (formation
window, holding period, favored group) with value, page, quote.

**6. Validate and adapt: the momentum ledger.** The check pass (the model
reads its extraction against the paper, reports contradictions with quotes:
zero for momentum). Then policy, not the model, maps paper to experiment and
labels every field: stated (six-month formation and holding, equal weights,
deciles), adapted (NYSE/AMEX breakpoints to our 415 names, long-short to
long-only, the $5 price floor dropped because snapshot prices are split
adjusted, the 1965–1997 sample to 2015–2025), assumed (252 days of history,
15 valid days in a formation month; these are engine parameters, counted
separately from the material fields, which is why the app shows "0 assumed").
Visual: the adaptation log as a four-color table.

**7. The calendar and the hygiene rules.** Warm-up 2012–2014, evaluation
2015-01-01 to 2025-12-31; signal from data through month t's last close,
trade at month t+1's first close, hold; six-month papers as six overlapping
cohorts; 10 bps per dollar traded with 0 and 25 reported; a name that leaves
the universe is sold at its last price. Enforced by mutation tests on the
engine and operations. The look-ahead we caught: a price floor applied to
split-adjusted prices silently dropped names such as Nvidia from the early
years; the independent verifier found it and the floor was removed. The
universe is 2025 membership with history back to 2012, a known survivorship
bias stated on every dossier. Course echoes: point-in-time discipline;
"contamination is the LLM-era look-ahead bias". Visual: timeline with the
monthly cycle zoomed.

**8. The scorecard as the course's factor checklist.** Session 3's six tests
for a real factor versus an artefact, each computed by WARREN: a prior story
(economic rationale, quoted), monotonicity of the ranked groups (Spearman
rho), t > 3, costs, out of sample, spanning. Momentum on our universe: decile
1 earns 1.27% a month, decile 10 earns 1.25%, spread t −0.07, rho −0.28: the
paper's actual claim, winners beat losers, does not reproduce here. Spanning
alpha t −0.17 with a 0.41 loading on the momentum factor: the strategy is
exposure, not alpha. Visual: the ranked-groups table and the alpha/spanning
table.

**9. Economics and the rubric.** 1,000 bootstrap paths, same dates for
strategy and benchmarks; $100M, 2/20, 5% hard hurdle, high-water mark, 10%
discount. Momentum: P(NPV>0) 76% raw, 48% demeaned, P(beats SPY) 11%;
research cost $19,500 of which model tokens $0.13. The rubric's four flags
with momentum's values: extraction checks green, performance positive,
robustness unknown, economics positive; first matching rule gives the verdict.
Visual: NPV histogram and the four flags.

**10. Verification.** Four independent checks, in the brief's words
"verified against ground truth": (1) an independent 30-line pandas
reimplementation of MAX matches the engine to 1.5e-15; (2) the two decoys
(gross profitability, time-series momentum) come back Unable to Evaluate with
the right missing inputs; (3) MAX and momentum rerun through the live API gave
backtest numbers identical to the build-night runs; (4) the fee engine
reproduces the original project's totals; 216 tests without network. Course
echoes: "treat the LLM as a model"; "failing loudly is a risk-management
property"; "any number you cannot reproduce is not a result". Visual: four
boxes with the evidence file for each.

**11. LLM versus numbers: what the model got right and wrong.** Right: every
momentum field quoted from the right page; the decoys refused. Wrong, and
caught: the oil mock paper carries an embedded instruction to declare itself
implementable; one of two extractions followed it, the check pass flagged the
contradiction, the dossier shows the low-reliability flag. Present oil as a
prompt-injection test, not as a result. And the interpretation scorecard:
momentum scores 36% against the preliminary reference specification; on
inspection the mismatches are in the reference, not the extraction, which is
itself a finding about how ground truth must be built. Visual: the oil
check-pass item beside the injected sentence; the momentum mismatch table.

**12. The nine papers.** Table: paper, adaptation in one phrase, verdict, net
CAGR vs equal weight, P(NPV>0): momentum 14.5% / 76% and oil 15.8% / 80%
Investigate Further; frog 11.9% / 43%, idiosyncratic volatility 11.9% / 35%,
52-week high 10.6% / 15%, seasonality 8.0% / 12%, MAX 6.3% / 2.5%
Deprioritize; two decoys Unable to Evaluate. Provenance footnote: extractions
were made by Claude sessions executing the exact prompts; MAX and momentum
were rerun through the API with identical results. Visual: table with badges.

**13. Reading the results like an analyst.** (a) Six of the seven runnable
signals use only the stock's own price history; the index shows their
excess-return correlations (52-week high with frog 0.77, with MAX 0.67); not
seven independent findings. (b) Multiple comparisons: seven papers tested, the
two leads have t-statistics near 1; the best of seven noise draws is about
1.35; nothing reaches the Harvey-Liu-Zhu bar of t > 3 and in 132 months
nothing can; Investigate Further means a lead. (c) Post-publication, equal
weight, long-only, large caps: weaker than the papers is the expected outcome.
Course echoes: the factor zoo; price factors are crowded. Visual: the
correlation block.

**14. The human-judgment ledger.** What people decided, not the model: the
registry of operations and the mapping rules; the adaptation policies and
defaults; the PARA exclusion and the Danaher patch; dropping the oil paper's
oracle timing signal; the rubric thresholds; the fee terms; every build-night
decision in `BUILD_LOG.md`. This is the brief's "where the document's
ambiguities required human judgment the code could not capture". Visual: a
two-column table, decision and reason.

**15. Honest limitations and next steps.** (Session 3 heads this section "do
not skip".) Survivor universe; long-only adaptations of long-short papers;
low power at 132 months; the narrative check is numeric only; reference
specifications are preliminary and must be rewritten by the group; the oil
paper's provenance; a rubric gap: robustness cannot currently score above
"unknown" because the sub-period test looks for a statistic the sub-period
entries do not carry, so Pursue is unreachable until fixed. Next: a
point-in-time universe, short legs with borrow costs, the group's own
reference specifications, the specification-error-cost study that was planned
and not built. Close on what WARREN is for: the first screen, so analyst time
goes to the papers that survive it.

## Appendix

**A1. The portal in 90 seconds.** One composite screenshot with numbered
arrows: tile and hover preview, Run, the nine steps live, the Summary and
quote line, chart, tabs (Dossier, Paper, Extraction, Evaluation, Economics),
previous runs. Or a live demo on the momentum run.

**A2. Parameters.** The read-only table: universe, window, warm-up, execution
convention, costs and grid, group floors, benchmarks, fund terms, Monte Carlo
settings, research-cost assumptions.

**A3. Dropping a PDF.** Read, extract, verdict, model summary; no backtest and
why; the Amihud example: "not implementable: needs daily dollar volume".

**A4. Backtest window and model.** The month slider and the orange window
chip; what changes; Sonnet default, Opus and Haiku call the API.

**A5. The API key.** Class key by default, personal key option, five-token
test before saving, stored in `.env` with a label, last four characters only.

**A6. The prompts.** The extraction system prompt's rules and the schema; the
check-pass prompt; the narrative prompt sees `results.json` only.

**A7. Reproducibility.** The GitHub address, the five commands to open
localhost, run folders, the cache, the MAX reimplementation test.

---

## Before the slides are built (from the review)

1. **Rubric fix.** Give the sub-period test an input it can evaluate (the
   relative t or sign of excess CAGR) so robustness can be moderate or strong
   and Pursue is reachable; rerun the demos. Say in slide 15 if not fixed.
2. **Narratives.** The committed demo summaries are now templates (commit
   3cc35a4). Regenerate all nine through the API so slides 3 and 12 show
   model-written text with real provenance, and print percentages rather than
   raw decimals in the narrative.
3. **Reference specification for momentum.** The group checks the paper's
   portfolio labels and filters against `papers/reference_specs/momentum.preliminary.json`
   before slide 11 claims the mismatches are the reference's fault.
4. **Pick one momentum source** for every screenshot: the API rerun. Do not
   mix "2 runs · 94%" from the demo with "3 runs · 85%" from the rerun.
