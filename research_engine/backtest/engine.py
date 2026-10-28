"""Share-based daily backtest engine.

Mechanics
- The book starts with value 1.0 in cash at the first trading day >= `start`.
- On each execution day (a key of `targets`, mapped to the first trading day on or after it)
  the book trades at that day's close to target shares = weight * V_before / price, where
  V_before is the marked value after any forced sales that day. On the first day, the most
  recent target dated before it (if any) is applied, so a benchmark keyed at the start of the
  data is bought on day one.
- Cost = cost_bps / 1e4 * |traded dollars|, paid from cash (cash absorbs it, so cash sits
  slightly below zero by the last cost between rebalances).
- Mark to market daily. A held ticker with no price on a day keeps its last price.
- Forced sale at the last available price, recorded with reason "forced_sale", when a held
  ticker's price series has ended (the first trading day after its last price in the
  snapshot) or when `eligible_fn(date)` (if given) no longer contains it.
- Valuation and trade prices are Yahoo adjusted closes (`ctx.adj_close`, SPY from
  `ctx.spy_adj_close`). Shares are therefore "adjusted-close units", not real share counts:
  this keeps the book's returns identical to adjusted-close total returns (dividends
  reinvested in the name), which is the convention every signal and benchmark uses.
- Gross and net books are simulated separately (cost 0 and cost_bps); holdings, trades and
  cash come from the net book.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

BENCH = "SPY"
_EPS = 1e-12


@dataclass
class BacktestResult:
    returns: pd.DataFrame            # daily: gross, net
    values: pd.DataFrame             # daily: gross_value, net_value, cash, n_holdings, turnover
    holdings: pd.DataFrame           # long: date, ticker, shares, price, value, weight (net book)
    trades: pd.DataFrame             # long: date, ticker, shares, price, value, cost, reason (net book)
    log: list[str] = field(default_factory=list)

    @property
    def forced_sales(self) -> pd.DataFrame:
        return self.trades[self.trades["reason"] == "forced_sale"]


def _price_panel(ctx, tickers: list[str]) -> pd.DataFrame:
    uni = [t for t in tickers if t != BENCH]
    missing = [t for t in uni if t not in ctx.adj_close.columns]
    if missing:
        raise KeyError(f"targets reference tickers not in the universe: {missing[:10]}")
    P = ctx.adj_close[uni]
    if BENCH in tickers:
        P = P.assign(**{BENCH: ctx.spy_adj_close})
    return P


def _simulate(P: np.ndarray, Pff: np.ndarray, ended: np.ndarray, days: pd.DatetimeIndex,
              tickers: np.ndarray, W: dict[int, np.ndarray], elig: Optional[np.ndarray],
              cost: float, record: bool):
    T, K = P.shape
    shares = np.zeros(K)
    cash = 1.0
    out_v = np.empty(T); out_cash = np.empty(T); out_n = np.empty(T, dtype=np.int64)
    out_turn = np.full(T, np.nan)
    hold_rows, trade_rows, log = [], [], []
    for t in range(T):
        px, lp = P[t], Pff[t]
        held = shares != 0
        # forced sales -------------------------------------------------------------
        force = held & ended[t]
        if elig is not None:
            force |= held & ~elig[t]
        if force.any():
            idx = np.where(force)[0]
            dollars = -shares[idx] * lp[idx]
            c = cost * np.abs(dollars)
            cash += -dollars.sum() - c.sum()
            if record:
                trade_rows.append((np.full(len(idx), t), idx, -shares[idx], lp[idx], dollars, c,
                                   np.ones(len(idx), dtype=bool)))
            for i in idx:
                log.append(f"{days[t].date()} forced_sale {tickers[i]} at {lp[i]:.6g} "
                           f"({'price series ended' if ended[t, i] else 'ineligible'})")
            shares[idx] = 0.0
        # rebalance ----------------------------------------------------------------
        w = W.get(t)
        if w is not None:
            w = w.copy()
            bad = (w > 0) & np.isnan(px)
            if elig is not None:
                bad |= (w > 0) & ~elig[t]
            if bad.any():
                for i in np.where(bad)[0]:
                    log.append(f"{days[t].date()} target_dropped {tickers[i]} (no price or ineligible)")
                w[bad] = 0.0
            tot = w.sum()
            if tot <= 0:
                log.append(f"{days[t].date()} target_skipped (no tradable names)")
            else:
                w /= tot
                hv = np.where(shares != 0, shares * lp, 0.0)
                V = cash + hv.sum()
                tp = np.where(np.isnan(px), lp, px)
                with np.errstate(invalid="ignore", divide="ignore"):
                    tgt = np.where(w > 0, w * V / px, 0.0)
                delta = tgt - shares
                dollars = delta * np.where(np.isnan(tp), 0.0, tp)
                tiny = np.abs(dollars) <= _EPS * abs(V)
                delta[tiny] = 0.0; dollars[tiny] = 0.0
                c = cost * np.abs(dollars)
                cash += -dollars.sum() - c.sum()
                traded = delta != 0
                shares[traded] = tgt[traded]
                out_turn[t] = 0.5 * np.abs(w - hv / V).sum()
                if record and traded.any():
                    idx = np.where(traded)[0]
                    trade_rows.append((np.full(len(idx), t), idx, delta[idx], tp[idx], dollars[idx], c[idx],
                                       np.zeros(len(idx), dtype=bool)))
        # mark to market -----------------------------------------------------------
        nz = shares != 0
        hv = shares[nz] * lp[nz]
        V = cash + hv.sum()
        out_v[t], out_cash[t], out_n[t] = V, cash, int(nz.sum())
        if record and nz.any():
            idx = np.where(nz)[0]
            hold_rows.append((np.full(len(idx), t), idx, shares[idx].copy(), lp[idx], hv, hv / V))
    return out_v, out_cash, out_n, out_turn, hold_rows, trade_rows, log


def _long(rows, days, tickers, names):
    if not rows:
        return pd.DataFrame({n: pd.Series(dtype="float64") for n in names})
    cols = [np.concatenate(c) for c in zip(*rows)]
    df = pd.DataFrame({"date": days[cols[0]], "ticker": tickers[cols[1]].astype(str)})
    for n, c in zip(names[2:], cols[2:]):
        df[n] = c
    return df


def run_backtest(ctx, targets: dict, cost_bps: float, start, end,
                 eligible_fn: Optional[Callable[[pd.Timestamp], pd.Index]] = None,
                 record: bool = True) -> BacktestResult:
    cal = ctx.calendar
    days = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    if len(days) == 0:
        raise ValueError("no trading days between start and end")
    # normalise target keys to trading days; first day carries the latest earlier target
    tg: dict[pd.Timestamp, pd.Series] = {}
    for k in sorted(targets):
        i = int(cal.searchsorted(pd.Timestamp(k), side="left"))
        if i < len(cal):
            tg[cal[i]] = targets[k]
    if days[0] not in tg:
        earlier = [k for k in tg if k < days[0]]
        if earlier:
            tg[days[0]] = tg[max(earlier)]
    tg = {k: v for k, v in tg.items() if days[0] <= k <= days[-1]}
    # Always book the whole universe (fixed column order), so floating-point sums up to any
    # date never depend on which names later targets reference (byte-identical look-ahead test).
    wanted = {str(t) for s in tg.values() for t in s.index[s.to_numpy() != 0]}
    tickers = [str(t) for t in ctx.tickers] + ([BENCH] if BENCH in wanted else [])
    unknown = sorted(wanted - set(tickers))
    if unknown:
        raise KeyError(f"targets reference tickers not in the universe: {unknown[:10]}")
    panel = _price_panel(ctx, tickers)
    full = panel.to_numpy(dtype=float)
    ff = panel.ffill().to_numpy(dtype=float)
    # last trading-day position with a price, per ticker
    has = ~np.isnan(full)
    last_pos = np.where(has.any(axis=0), len(cal) - 1 - np.argmax(has[::-1], axis=0), -1)
    pos0 = cal.get_loc(days[0])
    rows = np.arange(pos0, pos0 + len(days))
    P, Pff = full[rows], ff[rows]
    ended = rows[:, None] > last_pos[None, :]
    tick_arr = np.array(tickers, dtype=object)
    col = {t: i for i, t in enumerate(tickers)}
    W = {}
    for k, s in tg.items():
        w = np.zeros(len(tickers))
        s = s[s.to_numpy() != 0]
        w[[col[str(t)] for t in s.index]] = s.to_numpy(dtype=float)
        W[days.get_loc(k)] = w
    elig = None
    if eligible_fn is not None:
        elig = np.vstack([pd.Index(tickers).isin(eligible_fn(d)) for d in days])
        if BENCH in col:
            elig[:, col[BENCH]] = True

    c = cost_bps / 1e4
    net = _simulate(P, Pff, ended, days, tick_arr, W, elig, c, record)
    gross = net if c == 0 else _simulate(P, Pff, ended, days, tick_arr, W, elig, 0.0, False)
    v_net, cash, n_hold, turn, hold_rows, trade_rows, log = net
    v_gross = gross[0]
    prev = lambda v: np.concatenate([[1.0], v[:-1]])  # noqa: E731
    returns = pd.DataFrame({"gross": v_gross / prev(v_gross) - 1.0, "net": v_net / prev(v_net) - 1.0},
                           index=pd.DatetimeIndex(days, name="date"))
    values = pd.DataFrame({"gross_value": v_gross, "net_value": v_net, "cash": cash,
                           "n_holdings": n_hold, "turnover": turn}, index=returns.index)
    holdings = _long(hold_rows, days, tick_arr, ["date", "ticker", "shares", "price", "value", "weight"])
    trades = _long(trade_rows, days, tick_arr, ["date", "ticker", "shares", "price", "value", "cost", "reason"])
    if len(trades):
        trades["reason"] = np.where(trades["reason"].astype(bool), "forced_sale", "rebalance")
    else:
        trades["reason"] = pd.Series(dtype="str")
    return BacktestResult(returns=returns, values=values, holdings=holdings, trades=trades, log=log)
