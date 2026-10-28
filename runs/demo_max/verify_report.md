# Independent verification of runs/dev_max (MAX, decile 1)

Date: 2026-10-28. Verifier: independent agent (Claude), working from code and data only.
Test file: `tests/test_verify_max.py`. Rerun: `.venv/bin/python -m pytest tests/test_verify_max.py -s -q`
(15 tests, all passing; numbers are printed with the prefix `VERIFY`).

**Artifacts changed during verification.** `runs/dev_max` was rewritten at 04:53 while this
check was running. The earlier version had `price_floor: null` and included PARA. The current
version, which everything below verifies, has `price_floor: 5.0`, excludes PARA
(`contracts.EXCLUDE_TICKERS`) and patches DHR's adjusted close on 2016-07-05
(`contracts.ADJ_CLOSE_PATCHES`). The independent rebuild reads those two constants and applies
them with its own code. `run.log` holds both runs: 04:48 reports net CAGR 6.0% and 04:53 reports
6.4%. The stored `returns.parquet` matches 04:53, at 6.3565%.

| # | Check | Result | Key numbers |
|---|---|---|---|
| 1 | Signal by hand | PASS | 15 cases with \|diff\| = 0.0 (tolerance 1e-10) |
| 2 | Construction | PASS | all 132 months match exactly; one methodology issue (2b) |
| 3 | Independent returns | PASS | max \|diff\| 1.5e-15, median 2.6e-16 |
| 4 | Costs | PASS | cost = 10 bp × \|trade $\| to 1.7e-18 |
| 5 | Benchmarks | PASS | EW at 0 bp to 1.2e-15; SPY at 0 bp to 2.2e-16 |
| 6 | Look-ahead | PASS | holdings, signals, trades and returns up to 2020-06-30 identical |
| 6b | Reproducibility | PASS | stored files equal a fresh rerun of the stored spec |
| 7 | Sanity | PASS | vol and beta increase with every decile; turnover 0.795 |
| 8 | PARA | material | +0.27 pp to MAX net CAGR; the stored run already excludes PARA |

## 1. Independent signal
MAX was computed by hand from adj_close for AAPL, MSFT, XOM, JPM and NVDA on 2020-08-31,
2021-07-30 and 2024-06-28. These dates cover AAPL's 4:1 split on 2020-08-31 itself and NVDA's
splits on 2021-07-20 and 2024-06-10. All 15 values match `signals.parquet` exactly (diff 0.0).

## 2. Independent construction
Eligibility was rebuilt from the Parquet files and the constituents CSV with no engine code:
- adj_close present on the formation date;
- S&P add date <= the formation date;
- at least 252 non-missing adj_close days;
- close >= $5.

The independent decile 1 is the lowest ceil(n/10) names by MAX, with ties broken by ticker.

- Spot dates: 2015-06-30 (309 ranked, 31 in decile 1), 2020-03-31 (374, 38) and 2025-06-30
  (414, 42) match exactly.
- All 132 formation months match, both the decile-1 sets and the counts of ranked names.

**2b. Finding: the $5 floor uses Yahoo's back-split-adjusted close, which is a look-ahead.**
- Location: `research_engine/data/loaders.py:115` sets `self._close_np = self.close.to_numpy()`,
  and line 186 applies `m &= self._close_np[i] >= price_floor`. The docstring at line 193 says
  "raw close", but `close` in this snapshot is adjusted for later splits.
- Effect on eligibility: on Yahoo close the floor drops 50 eligible name-months, mostly NVDA in
  2014-2018 plus NFLX and CPRT. On a raw close rebuilt from the splits panel it drops only 3:
  FCX in 2016-01, and APA and F in 2020-03.
- Effect on decile 1: 9 months differ (2015-07, 2015-08, 2016-03, 2016-09, 2016-10, 2017-03,
  2017-09, 2017-12, 2019-01).
- Effect on CAGR when the floor uses raw close (a scratch rerun; no files were changed):

  | Series | Stored | Raw-close floor | Change |
  |---|---|---|---|
  | MAX net | 6.357% | 6.309% | -0.05 pp |
  | EW universe | 12.565% | 12.653% | +0.09 pp |
  | Decile 10 | 15.119% | 15.683% | +0.56 pp |

  The bias leaves winners that later split, NVDA above all, out of the high-MAX deciles.

## 3. Independent returns
A pandas backtest buys decile 1 at the execution-day close and holds it to the next execution
day. Each period's return is the mean of the members' adj_close returns over the period. The
backtest was compared with `returns_cost_0.parquet` `strategy_gross`, compounded over
(e_k, e_k+1], for 132 periods.
- Max \|diff\|: 1.47e-15. Median: 2.6e-16.

This confirms execution at the close of the first trading day, buy-and-hold within the month,
and no missing-price problems.

## 4. Costs
Over 132 execution days:
- Total cost per day equals 10 bp × sum \|trade value\|. Max error is 1.7e-18 per day and 0 per row.
- Cash moves by -(net trade $) - cost on each execution day, to 2.4e-16.
- Holdings value plus cash equals net_value, to 8.9e-16.
- On execution days, net minus gross return equals -cost / V_prev with a residual of at most
  7.8e-5 (median 7.3e-6). The mean drag is 15.8 bp per rebalance.

The residual has one source. The engine pays costs out of cash, so cash goes negative (down
to -0.4%) and the net book stays about 0.16% levered until the next rebalance. This is
documented at `engine.py:10` and lines 107-113. On non-execution days the net and gross
returns differ by at most 1.8e-4. The effect on CAGR is about 0.01 pp. This is minor and is
not a bug against the stated convention.

## 5. Benchmarks
- **EW universe, 0 bp.** The engine's targets were run at 0 bp. They match my equal-weighted
  mean of eligible names, rebalanced monthly: max \|diff\| 1.2e-15. At 10 bp
  (`returns.parquet`), the stored series trails mine by 5.2e-5 per month.
- **SPY, 0 bp.** The engine at 0 bp matches SPY adj_close buy-and-hold to 2.2e-16.
- **SPY, 10 bp, stored.** Day 1 returns -0.1%. After that, the 0.1% entry cost financed in cash
  makes daily returns differ by up to 7.6e-5. The terminal value is 3.99643, against
  3.99343 for (1 - 10 bp) × P_T/P_0, a 0.075% gap. This is the same cash-financing effect as
  in check 4.

## 6. Look-ahead
Method:
- Every adj_close and close after 2020-06-30 was multiplied by U(0.5, 1.5) noise, using
  `replace_prices` on a fresh DataContext.
- `run_dev` was rerun into a temporary directory, alongside a clean rerun.

Result:
- Up to 2020-06-30, the noisy rerun matches the clean one exactly in signals (22,965 rows),
  holdings (47,966), trades (4,081) and daily returns. After the cut they differ.
- 6b: the stored `runs/dev_max` files equal a fresh rerun of the stored spec exactly, for
  signals, holdings, trades, returns, returns_cost_0 and portfolio_value.

Caveat: this test cannot detect the floor look-ahead in 2b, because that look-ahead sits in
the snapshot's back-adjusted close itself.

## 7. Sanity
Values by decile, 1 to 10:

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Ann. vol | .151 | .157 | .165 | .169 | .178 | .188 | .199 | .210 | .222 | .246 |
| Beta vs SPY | 0.71 | 0.77 | 0.83 | 0.87 | 0.93 | 0.98 | 1.03 | 1.08 | 1.12 | 1.20 |

- Volatility rises with every decile (Spearman correlation 1.0).
- Decile 1's beta (0.71) is below decile 10's (1.20).
- Mean one-sided turnover is 0.795 per rebalance, or 0.793 including the initial buy.
- `group_1` equals `strategy_net` exactly.

## 8. PARA
`DataContext.load(exclude=...)` is supported. Both versions were rerun with the stored spec:

| Series | With PARA | Without PARA | Change |
|---|---|---|---|
| MAX net CAGR | 6.087% | 6.357% | +0.27 pp (above the 0.1 pp threshold) |
| EW universe | 12.545% | 12.565% | +0.02 pp |
| Decile 10 | 15.151% | 15.119% | -0.03 pp |

PARA sat in decile 1 for 20 months, the last on 2023-11-30. That includes the broken
adjusted-close move of -40.8% on 2023-12-15. Excluding PARA is therefore material. The stored
run is the PARA-excluded one, and its CAGR equals the without-PARA rerun exactly.

## Bugs and issues
1. **Look-ahead in the price floor.** It uses split-adjusted close (`loaders.py:115`, `:186`;
   the docstring at `:193` is wrong). Fix: apply the floor to a raw close, which is `close`
   multiplied by the product of splits after each date. Decile 1 differs in 9 months and
   decile 10's CAGR moves by 0.56 pp.
2. **Costs are financed with negative cash.** About 0.16% leverage remains between
   rebalances (`engine.py:107-113`), and the 10 bp SPY benchmark carries a permanent -0.1%
   cash position. The effect is about 0.01 pp of CAGR. This is cosmetic.
3. **run.log holds two runs** (6.0% and 6.4%). Only the second matches the stored returns.
