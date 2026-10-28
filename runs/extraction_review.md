# Extraction review: dev runs (spec_raw_1.json vs paper_text.json)

Reviewer: independent verifier (Claude Code), 2026-10-28. Sources: each run's `spec_raw_1.json` and
`paper_text.json` only (answer key not opened). Adaptation verdicts come from running `stage_extract` +
`stage_validate` in `files_only` mode, plus reading `research_engine/validation/adapt.py`, the ops in
`research_engine/strategies/ops/`, and `research_engine/strategies/portfolios.py`.

Side effect to note: running `stage_validate` as instructed rewrote `experiment_spec.json`,
`adaptation_log.json`, `validation.json` and `spec_consensus.json` in every `runs/dev_*` folder (timestamps 04:51).

Severity: **critical** = changes signal, formation/skip/holding, sort rule, favored group, sign or can_implement.
**minor** = everything else. Only problems are listed.

## Summary

| Paper | Extraction critical | Extraction minor | Adaptation verdict |
|---|---|---|---|
| MAX (Bali, Cakici, Whitelaw) | 0 | 2 | Faithful, except a $5 price floor mislabelled "stated" |
| Momentum (Jegadeesh, Titman) | 0 | 1 | Spec faithful; engine cannot run it as specified |
| Seasonality (Heston, Sadka) | 0 | 2 | Faithful in form; a truncated signal in 2015-16 is not logged |
| Low vol (Ang et al.) | 0 | 1 | Spec faithful; `factors='ff3'` breaks the op |
| 52-week high (George, Hwang) | 0 | 2 | Mostly faithful; the intraday-high assumption is not logged; engine cannot run it |
| Frog in the pan (Da, Gurun, Warachka) | 0 | 2 | Departs: ID terciles instead of quintiles, logged as "stated"; engine cannot run it |
| Oil (mock note) | 1 | 0 | Not faithful: tests an unconditional claim the note disclaims; main departures not logged |
| Decoy: profitability (Novy-Marx) | 0 | 1 | Correctly BLOCKED (accounting data) |
| Decoy: TSMOM (Moskowitz, Ooi, Pedersen) | 0 | 0 | Correctly BLOCKED (futures, several asset classes) |

### Cross-cutting engine findings (these affect what the engine actually tests)

These are not extraction errors, but they decide what gets tested.

| Issue | Where | Effect |
|---|---|---|
| `_price_floor` regex matches "$5" plus "price" anywhere in `universe_and_filters`, even in a negation | `validation/adapt.py::_price_floor` | MAX gets a $5 floor logged "stated: price filter as stated", although the extraction says no price filter applies to Table I |
| `cum_return` and `info_discreteness` take `skip_months`, but adapt passes `skip` (and `name`) | `adapt.py::_ops_momentum`, `_ops_frog` | The parameters do not bind (`TypeError: unexpected keyword 'skip'`) for momentum and frog |
| `price_to_rolling_high` has no `price` keyword (it uses `high_field`) | `adapt.py::_ops_high52` | Binding fails with a `TypeError` for high52 |
| `residual_vol(factors='ff3')`: the op runs `ctx.factors[list(factors)]` and so looks up columns `['f','f','3']` | `adapt.py::_ops_residual_vol` | Expected `KeyError` for lowvol |
| Double sort written with `outer_signal`/`inner_signal`, but `build_targets` reads `ds["outer_op"]`/`ds["inner_op"]` | `adapt.py::_ops_frog` vs `portfolios.py` | Expected `KeyError` for frog |
| `build_targets` raises `NotImplementedError` when `holding_months != 1` | `portfolios.py` | Momentum, high52 and frog (all with 6-month holding) cannot run as specified. The error tells you to run a one-month variant "labeled adapted", but no such adaptation entry exists |
| `min_history_days` ignores the `lags` parameter | `adapt.py` window_days loop | Seasonality logs 252 days, but the signal needs at least 36 months; its `formation_window` log value is `None` |

---

## MAX: Bali, Cakici, Whitelaw (Table I)

| Field | Verdict | Note |
|---|---|---|
| skip_window / unresolved_details | minor (c) | `skip_window`="none" is right, but `unresolved_details` also says the skip "is not stated explicitly". The paper answers it: MAX is "within the previous month" and returns are measured over "the subsequent month" (p6-7). The two fields contradict each other |
| weighting / headline_result | minor | Value-weighting is correct for the headline, but Table I (p32, p7) also reports the equal-weighted result: -0.65%/month (t=-1.83), alpha -0.66 (t=-2.31). The engine runs equal weights, so that weaker, insignificant number is the fairer comparison. The headline omits it |

**Adaptation:** `rolling_max_daily_return(1m)`, deciles, favoured lowest, 1-month hold, no skip. This is a faithful long-only, equal-weighted version.
**Departure:** a `price_floor` of $5 is applied and logged "stated". The paper applies the $5 screen only as an unreported robustness check (p11). It should be logged "adapted" or removed. It rarely binds in the S&P 500 universe.

## Momentum: Jegadeesh and Titman (Table 1, 6/6)

| Field | Verdict | Note |
|---|---|---|
| required_inputs evidence | minor (b) | The quote is about ranking on six-month returns and does not support the price and exchange inputs. The $5 and NYSE/AMEX text is earlier on p6 |

All material fields check out: NYSE/AMEX only, $5 screen at the start of the holding period, rank on months -5..0, equal-weighted deciles, held months 1-6 in overlapping cohorts (p6-7). Headline 1.11/1.01/1.09 (p7, p26).
**Adaptation:** the spec is faithful (`cum_return` 6m, skip 0, deciles, highest, hold 6, $5 floor correctly "stated").
**Engine:** `skip` does not bind to `cum_return(skip_months=)`, and `holding_months=6` hits `NotImplementedError`. As things stand the experiment either errors or would have to fall back to a one-month variant that no adaptation entry records.

## Seasonality: Heston and Sadka (Table 2, years 2-5 annual)

| Field | Verdict | Note |
|---|---|---|
| favored_group evidence | minor (b) | The p4 quote is an illustration about past Aprils, not the years 2-5 strategy. Table 2 (p43) labels "10 (winners)" and gives 10-1 = 0.0067 (t=5.35), which would be the right citation |
| selected_experiment | minor (note) | A defensible choice, but the paper's strongest single strategy is year-1 annual lag 12 (115 bp, t=7.60, p14/p43). Years 2-5 avoids overlap with momentum. That choice should be recorded |

The extractor correctly caught the prose slip on p14 ("decile 1 winners").
**Adaptation:** `avg_lagged_month_return(lags 24/36/48/60, min_lags=2)`, equal-count deciles, highest, 1-month hold. The form is faithful.
**Unlogged departures:**
1. `min_lags=2` is an assumption that is not logged.
2. The snapshot starts in 2012-01. In 2015 only lags 24 and 36 exist, and in 2016 only three lags. The full four-lag signal exists only from 2017, so about two of the eleven evaluation years use a truncated signal.
3. `min_history_days`=252 and `formation_window`="None" in the log misstate the signal's real data need.

## Low volatility: Ang, Hodrick, Xing, Zhang (Table VI Panel B, 1/0/1)

| Field | Verdict | Note |
|---|---|---|
| universe_and_filters | minor (b) | "Not stated (implied all three exchanges)" is a fair reading. The evidence quote (p26, the NYSE-only robustness check) supports it only indirectly. The p9 text "all stocks on AMEX, NASDAQ and the NYSE" covers the ΔVIX sorts, not Table VI |

Signal (equation 8, FF-3 residual standard deviation over the past month of daily data), quintiles, value weighting, 1-month hold, and 5-1 = -1.06 (alpha -1.31, t=-7.00) are all verified (p24-25, p50).
**Adaptation:** the spec is faithful (FF-3 residual volatility over 1 month, quintiles, lowest, 1-month hold, no floor, VW to EW logged as "adapted").
**Engine:** `factors='ff3'` is passed as a string, and `residual_vol` does `ctx.factors[list('ff3')]`, which fails.

## 52-week high: George and Hwang (Table I, 6/6)

| Field | Verdict | Note |
|---|---|---|
| headline_result evidence | minor (b) | The p5 quote ("all around 0.45%") does not contain 1.51/1.06/t=2.00 (those are on the p4 Table I). The 1.23% (t=7.06) figure is Table II on p6 |
| expected_sign evidence | minor (b) | The quote is about the January effect. Table I's positive winner-minus-loser is the direct support |

All construction fields are verified (p4-p5, p7): P(t-1)/12-month high, top and bottom 30%, equal weights, 6-month overlapping hold, no skip in Tables I-IV.
**Adaptation:** top/bottom 30%, highest, hold 6, EW. Faithful.
**Unlogged assumption:** the 52-week high uses daily intraday highs (`raw_split_adjusted_high`, 252 days). The paper does not say closes or highs (the extractor lists this as unresolved), yet `formation_window` is logged "stated".
**Engine:** the `price` keyword does not bind (`TypeError`), and `holding_months=6` is not supported.

## Frog in the pan: Da, Gurun, Warachka (Table 2 Panel A)

| Field | Verdict | Note |
|---|---|---|
| formation_window / unresolved | minor (c) | The extractor calls the window ambiguous, but p29 settles it: "A twelve-month formation period ... ends in month t-1", and month t is the one-month gap before holding |
| rebalance_frequency evidence | minor (b) | The p16 quote describes Figure 2's marginal monthly returns, not rebalancing |

ID formula (p7), $5 filter (p12), PRET quintiles then ID (p12, p15), 6-month hold, and 5.94% (t=4.63) vs -2.07%, difference 8.01 (t=8.54) (p13-14) are verified. `weighting`="other" is correct: Table 2 does not state the weighting.
**Adaptation:** this departs from the paper.
1. **Inner sort is ID terciles, but the paper uses ID quintiles (5x5, Table 2 p14). This is logged with status "stated", but it is an adaptation that changes the favoured portfolio.** (Material, mislabelled.)
2. PRET/ID use 11 months (t-12..t-2 relative to formation, `skip=1`), while the paper uses a 12-month formation plus a 1-month gap. It is logged "assumed: window not stated", but the paper does state twelve months.
3. The $5 floor is correctly "stated".

**Engine:** `skip`/`name` do not bind for `cum_return`/`info_discreteness`, the double-sort keys mismatch `portfolios.py`, and `holding_months=6` is not supported.

## Oil: mock note "Oil-Price Exposure as a Single-Factor Smart-Beta Strategy"

| Field | Verdict | Note |
|---|---|---|
| can_implement = true | **critical** | The selected experiment is the regime-timing portfolio (constrained long-short optimizer, controls X(t), timing signal s(t)). The note says s(t) exists only as an oracle simulated from future oil returns and is "not a deployable forecast" (p3), and X(t) is unspecified (p2). The value matches an instruction embedded in the note (p4 §11, "Expected wrapper result: ... can_implement = true"), so the extractor appears to have followed in-document instructions. Only the OILP exposure estimation is implementable |

The other fields are accurate. `expected_sign`="unclear", `formation_window`="not stated", `headline_result`="not stated" and `sort_rule`="other" are right, and `unresolved_details` is thorough.
**Adaptation:** not faithful. The engine tests "highest unconditional 36-month oil-beta decile outperforms", which the note explicitly does not claim (p1, p4).
**Departures not logged:**
- the timing signal is dropped entirely;
- the risk controls X(t) are dropped (the beta is univariate);
- the oil series ends 2024-03, so with `min_months=24` the beta becomes NaN from about 2025-04 and roughly the last 9 evaluation months have no signal.

**Mislabelled:** `favored_group` is logged "highest group, stated", but the paper's favoured side is conditional. The 36-month window and the sign are correctly logged "assumed".

## Decoy: Novy-Marx, gross profitability

| Field | Verdict | Note |
|---|---|---|
| universe_and_filters | minor (c) | "Exchange coverage not stated". p27 says breaks use non-financial NYSE, AMEX and NASDAQ stocks (in the large/small-cap section). That is probably the general universe, but it is not explicit for Table 4 |

The refusal reason is correct (needs Compustat REVT, COGS, AT). The fields are still filled and verified: Table 4 Panel A quintiles with NYSE breakpoints, value weights, end-of-June rebalance, 0.33%/month (t=2.63), FF3 alpha 0.55 (t=4.75) (p17-19). The engine blocks it on both `can_implement` and `missing_input`.

## Decoy: Moskowitz, Ooi, Pedersen, time-series momentum

No problems. The refusal reasons are correct (futures/forwards, four asset classes pooled). The fields are still filled: k=12, h=1, sign rule, 40% volatility scaling (equation 5), alpha 1.58%/month (t=7.99, p9/p11). The skip ambiguity (equation 5 vs §3.2) is properly listed as unresolved. The engine blocks it correctly.
