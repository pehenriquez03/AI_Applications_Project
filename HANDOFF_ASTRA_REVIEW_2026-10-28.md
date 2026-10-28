# Review handoff for Astra — grade WARREN as the professor would

Written 2026-10-28 by Claude for Philip Henriquez (IEOR 4723, Group 6).
Repository: https://github.com/pehenriquez03/AI_Applications_Project, branch `main`
(identical to `engine`). Start with `README.md`; it is the lead document.

## What you are reviewing

WARREN, "your research screening agent": a paper goes in, a research dossier
comes out with one of four verdicts (Pursue, Investigate Further, Deprioritize,
Unable to Evaluate). Claude Sonnet 5.5 reads the paper (page-cited extraction,
a check pass against the paper, the dossier prose); Python does every number
(eligibility, signals, long-only equal-weight monthly sorts on 415 S&P 500
names, holdings-level backtest 2015–2025 with costs, scorecards, block-bootstrap
fund economics with 2/20 fees). Nine papers are built in: six core (MAX,
momentum, seasonality, idiosyncratic volatility, 52-week high, frog in the
pan), the group's mock oil-factor paper, and two decoys (Novy-Marx 2013,
Moskowitz-Ooi-Pedersen 2012) that must come back "Unable to Evaluate". Any
other PDF can be dropped: it is extracted and summarized but not backtested.

## The professor's yardstick

The course brief (`group project-261009.pdf`) asks for an end-to-end solution
that uses the LLM where it adds value. Its examples 1 and 3 are the closest
templates, and they imply the criteria below. Grade against them, not against
what you would have built.

1. **Document to executable logic.** Does the model turn prose into a
   specification precise enough to run, and is that step tested rather than
   assumed? Evidence: `research_engine/contracts.py` (StrategySpec),
   `research_engine/extraction/`, `papers/answer_key.json` (never in a prompt),
   `scorecard_interpretation.json` in each run, the Extraction tab.
2. **Verified against ground truth.** Are the generated specifications and the
   engine checked against something independent? Evidence: `tests/test_verify_max.py`
   (independent pandas reimplementation agrees to 1.5e-15), the reference
   specifications in `papers/reference_specs/` (currently preliminary, derived
   from the answer key; the group must write its own), the two decoys, the
   check pass with contradicting quotes.
3. **Where human judgment was needed.** Is there an explicit ledger of what the
   paper left unresolved and what policy filled the gap? Evidence:
   `adaptation_log.json` in every run (stated / assumed / adapted / unresolved
   per field), the "Unresolved details" list in the Extraction tab,
   `BUILD_LOG.md`, the defaults table in `BUILD_PLAN_FINAL.md`.
4. **Backtesting hygiene.** Look-ahead, survivorship, multiple comparisons,
   cost realism. Evidence: signals use the prior month's last close and trade
   at the next first close (`research_engine/backtest/engine.py`); survivor
   universe is stated as a limitation in every dossier; specifications are fixed
   from the paper before any result is seen; five past-return papers are shown
   with their excess-return correlation in `runs/index.html`; cost grid 0/10/25
   bps; no price floors on split-adjusted prices (a look-ahead that was found
   and removed, see `BUILD_LOG.md`).
5. **How much comes from the LLM versus the numbers.** The dossier's verdict is
   produced by a fixed rubric from the scorecards (`research_engine/reporting/rubric.py`);
   the model writes the prose afterwards and every number it cites is checked
   (`narrative.json: verified, unverified_numbers`). Judge whether that division
   is real and whether the interpretation scorecard measures the LLM's
   contribution honestly.
6. **A working prototype on public data.** Yahoo prices, Ken French factors,
   free paper copies; a local Streamlit app; 210 tests with no network.

## How to run it (read-only review)

```bash
git clone https://github.com/pehenriquez03/AI_Applications_Project && cd AI_Applications_Project
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q                      # 210 tests, no model calls, no network
.venv/bin/streamlit run research_engine/app/streamlit_app.py
```

The nine demo runs under `runs/demo_*` are committed with their dossiers and
open in the app without a key (choose "Continue without a key"). Rerunning a
paper needs the PDFs in `papers/pdf/` (listed in `papers/README.md`; not
redistributed), the data snapshot (`research_engine/data/snapshot.py` rebuilds
it), and an Anthropic key. Do not make model calls for the review; everything
you need is on disk. Never put `papers/answer_key.json` into a prompt.

## Results to check, not take on trust

| Paper | Verdict | Net CAGR vs EW universe | Where |
|---|---|---|---|
| Momentum (6-month cohorts) | Investigate Further | 14.5% vs 12.7%, P(NPV>0) 76% | `runs/demo_momentum/dossier.html` |
| Oil mock | Investigate Further | 15.8% vs 12.8%, extraction disputed | `runs/demo_oil/` |
| MAX | Deprioritize | 6.3% | `runs/demo_max/` |
| Seasonality | Deprioritize | 8.0% | `runs/demo_seasonality/` |
| 52-week high | Deprioritize | 10.6% | `runs/demo_high52/` |
| Idiosyncratic volatility | Deprioritize | 11.9% | `runs/demo_lowvol/` |
| Frog in the pan | Deprioritize | 11.9% | `runs/demo_frog/` |
| Novy-Marx, TSMOM | Unable to Evaluate | blocked on missing inputs | `runs/demo_decoy_*/` |

MAX and momentum were rerun through the live API on 2026-10-28 with the class
key and reproduced these numbers exactly (`cache/llm/` entries with
`"source": "api"`); the other demo runs still carry model answers produced by
Claude sessions executing the same prompts during the build night
(`llm_usage.json: "source": "session"`). Say whether that provenance is
acceptable for a submission or must be replaced by API runs.

## What the group already knows is weak

- Survivor universe (today's members with history back to 2012).
- Reference specifications for the interpretation score are preliminary and
  derived from the answer key; the group has not yet written its own.
- Long-only adaptations of long-short papers; equal weights throughout.
- 132 months of post-publication data cannot reach t > 3 for realistic spreads.
- The oil mock paper contains an embedded instruction telling the model the
  expected answer (inherited from the original project); the pipeline flags
  the disagreement between its extraction runs rather than hiding it.
- Effort saved versus an analyst is asserted, not measured (the group chose
  not to run a human baseline).
- The app uses the key shared in class by default; `.env` is gitignored and the
  app binds to localhost only.

## What the review should return

A written review, severity-ranked, that Philip can take to the group:

1. **Verdict on fit.** Does this meet the brief as examples 1 and 3 define it?
   Where does it fall short of "verified against ground truth" and "explicit
   ledger of human judgment"?
2. **Findings.** For each: file and line or run folder, what is wrong, a
   concrete failure or misleading claim, and the fix. Separate what you
   verified by reading or running from what you suspect.
3. **The three things to fix before submission**, in order.
4. **An estimated grade** on the course scale, with the sentence you think the
   professor would write.
5. **Questions the group must answer in the presentation**, the ones a
   skeptical reader asks first (for example: why would an analyst trust a
   verdict produced from a survivor universe; what does the interpretation
   score mean while the reference specs are preliminary; what did the model
   get wrong on the oil paper and how was it caught).

Read-only: do not modify the repository. If something cannot be verified from
the files, say so rather than guessing. Earlier context, if you want it:
`PLAN_HANDOFF_2026-10-28.md` (the design the group agreed), `BUILD_PLAN_FINAL.md`
(the plan as built), `BUILD_LOG.md` (every decision taken during the build).
