# Build log, branch `engine`, night of 2026-10-28

Decisions taken without Philip, in order. Times are local (America/New_York).

- 04:10 Snapshot built: 415 tickers with full Yahoo data 2012-2025; 19 excluded
  (no Yahoo data after batch download and two single-ticker retries;
  several are post-2024 delistings). Excluded per Philip's instruction.
- 04:35 API smoke test: the key in `.env` is rejected (401). Decision: build
  the full API path, test with fixtures, and produce real extractions via
  "session" execution (a subagent executes the exact prompt and the result
  is imported into the cache with source="session"). Tomorrow's key makes
  the same prompts run through the API with no other change.
- 04:40 Contracts written (research_engine/contracts.py, CONTRACTS.md).
  Wave 1 launched: A (Opus) data/strategies/backtest, B (Opus)
  extraction/validation, C (Sonnet) economics. Pipeline orchestrator and CLI
  written by the orchestrator.
- 04:50 Wave 2 launched early against the contracts with synthetic fixtures:
  D (Sonnet) evaluation, E (Opus) reporting, F (Sonnet) app.
- 05:05 Agent C done: economics, 28 tests pass; legacy fee totals reproduced
  to 1e-6 (run 1: 35440.923). Vectorized fee path proven equal to the legacy
  loop on 20 random paths. Monte Carlo 1000 paths in 0.17 s.
- 05:40 Agent B done: documents/extraction/validation, 40 tests. API 401
  confirmed. Nine papers ingested; session prompts exported. Nine session
  extraction agents launched (Opus for the seven cores, Sonnet for decoys).
- 05:45 Agent F done: Streamlit app with mock run, 5 smoke tests. Agent D
  done: evaluation, 25 tests; first MAX numbers (unverified): net CAGR 6.0%
  vs SPY 13.5% and EW 12.7%; decile pattern opposite to the paper (rho +0.84).
  Pipeline fixed: resume guards for stages 4-9, validation.json constant,
  mock runs whitelisted in .gitignore.
- 06:00 Agent A done (engine, 40 tests; MAX hand checks to 1e-10). Agent E
  done (reporting, 17 tests). Data verifier: exclude PARA only (Yahoo's
  "PARA" is not Paramount: 0-75 shares/day, $100 "dividends"); DHR
  2016-07-05 is a Yahoo double-applied spin-off -> patched to the close
  return (contracts.ADJ_CLOSE_PATCHES); the 71 "mismatch" tickers are 1e-6
  rounding noise; Yahoo close/high/low are already split-adjusted, so the
  split rebuild is a no-op and price floors apply to split-adjusted prices.
- 06:10 Checkpoint 2 committed. Pipeline runs end to end: MAX deprioritize
  (net CAGR 6.4% vs EW 12.6%, P(NPV>0) 3%), seasonality (lags 24-60)
  deprioritize, oil investigate further, both decoys unable to evaluate
  (blocked at validation with the right reasons). Lowvol fixed (factor
  alias) and rerun.
- 06:30 Cohorts done (Opus): K=6 overlapping cohorts, 1/K per cohort netted
  before trading; steady-state turnover 15.3%/month vs 1/K = 16.7%. Ops
  agent (Sonnet): avg_lagged_month_return (Heston-Sadka annual lags) and
  oil_beta (36-month monthly regression on Aidan's oil series, which ends
  2024-03). Extraction re-read verifier: 0 critical problems in 8 of 9
  extractions; oil's can_implement=true coincides with the instruction
  embedded in the mock paper; adaptation labels fixed (frog inner terciles
  = adapted; oil favored group = assumed; timing signal dropped = adapted;
  min_lags = assumed). MAX engine verifier: 8/8 checks pass, engine vs
  independent pandas backtest to 1.5e-15; found the price-floor look-ahead
  (floor applied to split-adjusted prices) -> decision: no price floors at
  all, logged as adapted when a paper states one.
- 06:45 Adapter parameter names aligned with the ops (skip_months,
  high_field, pret_12_2/info_discreteness, outer_op/inner_op). All seven
  implementable papers run end to end; check passes: 6 agree, oil disputes
  can_implement (recorded as a critical disagreement). Preliminary reference
  specs derived from the answer key (used only by the scorer); scorer made
  paraphrase-tolerant (containment / token Jaccard >= 0.5 / month parsing);
  MAX agreement 0.69 vs exact-match 0.375.
- 07:20 All seven implementable papers run end to end with check passes and
  preliminary reference specs; narratives written by session agents for all
  seven (every number verified against results.json); narrative writer now
  reads the session import file directly. Second extraction run launched for
  run-to-run agreement. Compare page gains the cross-strategy monthly
  correlation table. App integration on real runs in progress.
- 07:50 Second extraction run imported for all seven papers; consensus
  agreement (paraphrase-tolerant) 89-100% on six papers, 75% on the oil mock
  (run 2 says cannot implement; check pass disputes run 1). App integrated on
  the real runs (dedicated tables, 9 smoke tests). Momentum narrative
  regenerated after the research-cost figure changed with run-2 token usage.
- 08:15 Final: nine demo runs under runs/demo_* (seven dossiers with
  model-written, number-verified narratives; two decoys blocked), compare
  page with excess-return correlations, app verified on the real runs,
  193 tests passing. Research cost excludes the narrative call's own usage.
  Committed as checkpoint 3 on branch engine. Not pushed.

## Open items for Philip

1. API key: put a working `ANTHROPIC_API_KEY` in `.env`; the same prompts
   then run through the API (cache keeps the demo instant). Click
   "Regenerate narrative" on any run to see a live model call.
2. Reference specifications: the group writes one per paper
   (`papers/reference_specs/`), replacing the preliminary ones derived from
   the answer key; the interpretation scorecards then lose the preliminary label.
3. Known approximations: cohorts reset to 1/K each month (turnover effect
   measured at +0.7 pp/month); costs are paid from cash so the book sits
   ~0.16% levered between rebalances (~0.01 pp CAGR); the oil series ends
   2024-03 so the oil run's last months are skipped; the oil mock's
   implementability is disputed by one of two extractions and by the check pass.
