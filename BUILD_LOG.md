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
