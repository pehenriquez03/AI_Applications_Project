"""Load the frozen snapshot into one in-memory DataContext.

Conventions (see CONTRACTS.md and BUILD_PLAN_FINAL.md):
- Every panel is wide: index = trading dates (datetime64[ns]), columns = tickers.
- SPY is removed from the universe panels and kept as `spy_adj_close`, `spy_close`,
  `spy_returns`.
- `returns` are simple daily returns from Yahoo adjusted close (no fill across gaps).
- `calendar` is the snapshot's own trading-day index (NYSE days: SPY trades on every row).

Data finding (2026-10-28, Agent A): Yahoo's `close`/`high`/`low` in this snapshot are
already split-adjusted (yfinance auto_adjust=False still back-adjusts splits; only
dividends are left out). `raw_split_adjusted` therefore applies a split factor only
where the split is visible in the close series itself (the close drops by about the
split ratio on the split date); for this snapshot that leaves the series unchanged,
which is the correct split-only adjusted series.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import json
import numpy as np
import pandas as pd

from research_engine.contracts import MIN_HISTORY_DAYS, SNAPSHOT_DIR, UNIVERSE_FILE

PRICE_FIELDS = ("open", "high", "low", "close", "adj_close", "volume", "dividends", "splits")
BENCH = "SPY"
MIN_DETECTABLE_SPLIT = 1.2


def _ns(df: pd.DataFrame | pd.Series):
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index).as_unit("ns")
    df.index.name = "date"
    return df


def visible_split_factors(close: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Split ratios (date x ticker, 1.0 elsewhere) for splits the close series actually shows.

    A split of ratio r on day t is "visible" when the close-to-close move x = c_t / c_{t-1}
    is closer to 1/r than to 1 in log terms, i.e. |log(x*r)| < |log(x)|. Ratios within
    MIN_DETECTABLE_SPLIT of 1 (Yahoo records some spin-offs as 1.003-1.03 "splits") cannot be
    told apart from an ordinary daily move and are never applied.
    """
    s = splits.reindex(index=close.index, columns=close.columns).fillna(0.0)
    s = s.where(s > 0, 1.0)
    x = close / close.ffill().shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lx = np.log(x.to_numpy())
        lxr = np.log((x * s).to_numpy())
    big = np.abs(np.log(s.to_numpy())) >= np.log(MIN_DETECTABLE_SPLIT)
    vis = big & np.isfinite(lx) & np.isfinite(lxr) & (np.abs(lxr) < np.abs(lx))
    return pd.DataFrame(np.where(vis, s.to_numpy(), 1.0), index=close.index, columns=close.columns)


def split_adjust(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Divide every price before a split date by the product of later split ratios."""
    f = factors.reindex(index=prices.index, columns=prices.columns).fillna(1.0)
    # cumulative factor of splits strictly after each date
    after = f.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
    return prices / after


class DataContext:
    """All market data a run needs. Build with `DataContext.load()`."""

    def __init__(self, panels: dict[str, pd.DataFrame], ff3: pd.DataFrame, mom: pd.DataFrame,
                 add_dates: pd.Series, snapshot_dir: Optional[Path] = None):
        self.snapshot_dir = snapshot_dir
        adj = _ns(panels["adj_close"])
        self.calendar: pd.DatetimeIndex = adj.index
        universe = [c for c in adj.columns if c != BENCH]
        self.tickers = pd.Index(universe, name="ticker")

        def uni(name):
            p = _ns(panels[name]).reindex(index=self.calendar)
            return p.reindex(columns=self.tickers)

        self.adj_close = uni("adj_close")
        self.close = uni("close")
        self.high = uni("high")
        self.low = uni("low")
        self.open = uni("open") if "open" in panels else None
        self.volume = uni("volume")
        self.splits = uni("splits").fillna(0.0)
        self.dividends = uni("dividends").fillna(0.0) if "dividends" in panels else None

        self.spy_adj_close = adj[BENCH].copy()
        self.spy_close = _ns(panels["close"])[BENCH].reindex(self.calendar)
        self.returns = self.adj_close.pct_change(fill_method=None)
        self.spy_returns = self.spy_adj_close.pct_change(fill_method=None)
        self.spy_returns.name = BENCH

        f = _ns(ff3).rename(columns={"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml", "RF": "rf"})
        m = _ns(mom).rename(columns={"Mom": "mom"})
        self.factors = pd.concat([f[["mkt_rf", "smb", "hml"]], m[["mom"]], f[["rf"]]], axis=1).reindex(self.calendar)

        cal = pd.Series(self.calendar, index=self.calendar)
        per = self.calendar.to_period("M")
        self.month_ends = pd.DatetimeIndex(cal.groupby(per).max().to_numpy(), name="date")
        self.first_days = pd.DatetimeIndex(cal.groupby(per).min().to_numpy(), name="date")

        self.add_dates = add_dates.reindex(self.tickers)
        self.add_dates.name = "add_date"

        # precomputed arrays for fast eligibility
        self._has_px = self.adj_close.notna().to_numpy()
        self._hist = np.cumsum(self._has_px, axis=0)
        ad = self.add_dates.to_numpy(dtype="datetime64[ns]")
        self._add_ns = np.where(np.isnat(ad), np.iinfo(np.int64).min, ad.astype("int64"))
        self._close_np = self.close.to_numpy()
        self._cal_pos = {d: i for i, d in enumerate(self.calendar)}
        self._split_factors = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, snapshot_dir: Path | str = SNAPSHOT_DIR, universe_file: Path | str = UNIVERSE_FILE,
             fresh: bool = False, exclude: tuple[str, ...] | None = None) -> "DataContext":
        """Read the snapshot. Cached per (paths, exclude); fresh=True gives an independent copy.
        `exclude` drops tickers from the universe (e.g. a data-quality exclusion)."""
        if exclude is None:
            from ..contracts import EXCLUDE_TICKERS
            exclude = EXCLUDE_TICKERS
        exclude = tuple(sorted(exclude))
        if fresh:
            return cls._read(str(snapshot_dir), str(universe_file), exclude)
        return _cached_load(str(snapshot_dir), str(universe_file), exclude)

    @classmethod
    def _read(cls, snapshot_dir: str, universe_file: str, exclude: tuple[str, ...] = ()) -> "DataContext":
        d = Path(snapshot_dir)
        panels = {f: pd.read_parquet(d / f"{f}.parquet") for f in PRICE_FIELDS}
        manifest = json.loads((d / "manifest.json").read_text())
        keep = [t for t in manifest["universe_tickers"] if t not in set(exclude)] + [BENCH]
        panels = {k: v[[c for c in v.columns if c in keep]] for k, v in panels.items()}
        # Single-day adjusted-close corrections (contracts.ADJ_CLOSE_PATCHES): rescale the adjusted
        # series before the bad day so that day's adjusted return equals the close return.
        from ..contracts import ADJ_CLOSE_PATCHES
        adj, clo = panels["adj_close"], panels["close"]
        for (tk, day), _reason in ADJ_CLOSE_PATCHES.items():
            if tk in adj.columns:
                t = pd.Timestamp(day); i = adj.index.get_loc(t); prev = adj.index[i - 1]
                bad = adj.at[t, tk] / adj.at[prev, tk]; good = clo.at[t, tk] / clo.at[prev, tk]
                adj.loc[:prev, tk] = adj.loc[:prev, tk] * (bad / good)
        panels["adj_close"] = adj
        ff3 = pd.read_parquet(d / "ff3_daily.parquet")
        mom = pd.read_parquet(d / "mom_daily.parquet")
        add_dates = read_add_dates(Path(universe_file))
        return cls(panels, ff3, mom, add_dates, snapshot_dir=d)

    def replace_prices(self, **panels: pd.DataFrame) -> "DataContext":
        """New context with some universe panels swapped (used by look-ahead tests)."""
        base = {"adj_close": self.adj_close, "close": self.close, "high": self.high, "low": self.low,
                "open": self.open, "volume": self.volume, "splits": self.splits, "dividends": self.dividends}
        base.update(panels)
        spy = {"adj_close": self.spy_adj_close, "close": self.spy_close}
        full = {}
        for k, v in base.items():
            v = v.copy()
            if k in spy:
                v[BENCH] = spy[k]
            elif BENCH not in v.columns:
                v[BENCH] = np.nan
            full[k] = v
        ff3 = self.factors[["mkt_rf", "smb", "hml", "rf"]].rename(
            columns={"mkt_rf": "Mkt-RF", "smb": "SMB", "hml": "HML", "rf": "RF"})
        mom = self.factors[["mom"]].rename(columns={"mom": "Mom"})
        return DataContext(full, ff3, mom, self.add_dates.copy(), snapshot_dir=self.snapshot_dir)

    # ------------------------------------------------------------- eligibility
    def eligible_mask(self, date, min_history_days: int = MIN_HISTORY_DAYS,
                      price_floor: Optional[float] = None) -> np.ndarray:
        i = self._cal_pos.get(pd.Timestamp(date))
        if i is None:  # not a trading day: use the last trading day before it
            i = int(self.calendar.searchsorted(pd.Timestamp(date), side="right")) - 1
            if i < 0:
                return np.zeros(len(self.tickers), dtype=bool)
        d_ns = pd.Timestamp(date).as_unit("ns").value
        m = self._has_px[i] & (self._add_ns <= d_ns) & (self._hist[i] >= min_history_days)
        if price_floor is not None:
            with np.errstate(invalid="ignore"):
                m &= self._close_np[i] >= price_floor
        return m

    def eligible(self, date, min_history_days: int = MIN_HISTORY_DAYS,
                 price_floor: Optional[float] = None) -> pd.Index:
        """Tickers eligible at `date`: priced that day, added to the S&P 500 by then
        (NaT = always), >= min_history_days non-missing adj_close up to and including
        `date`, and raw close >= price_floor when a floor is given."""
        return self.tickers[self.eligible_mask(date, min_history_days, price_floor)]

    def eligibility_panel(self, dates, min_history_days: int = MIN_HISTORY_DAYS,
                          price_floor: Optional[float] = None) -> pd.DataFrame:
        rows = [self.eligible_mask(d, min_history_days, price_floor) for d in dates]
        return pd.DataFrame(np.array(rows, dtype=bool), index=pd.DatetimeIndex(dates, name="date"),
                            columns=self.tickers)

    # ------------------------------------------------------------- split rebuild
    def raw_split_adjusted(self, field: str = "close") -> pd.DataFrame:
        """close/high/low adjusted for splits only (no dividends), rebuilt from the splits panel.
        Splits are detected on the close series and applied to the requested field."""
        if field not in ("close", "high", "low", "open"):
            raise ValueError(f"raw_split_adjusted supports close/high/low/open, not {field!r}")
        return split_adjust(getattr(self, field), self.split_factors())

    def split_factors(self) -> pd.DataFrame:
        """Split ratio on each split day that the close series shows (1.0 elsewhere).
        Detection on day t uses only the close on t and the prior close, plus the split record."""
        if self._split_factors is None:
            self._split_factors = visible_split_factors(self.close, self.splits)
        return self._split_factors

    # ----------------------------------------------------------------- helpers
    def next_trading_day(self, date) -> Optional[pd.Timestamp]:
        i = int(self.calendar.searchsorted(pd.Timestamp(date), side="right"))
        return self.calendar[i] if i < len(self.calendar) else None

    def execution_day(self, month_end) -> Optional[pd.Timestamp]:
        """First trading day after a formation month-end (a member of first_days)."""
        return self.next_trading_day(month_end)


def read_add_dates(universe_file: Path) -> pd.Series:
    c = pd.read_csv(universe_file)
    t = c["Symbol"].astype(str).str.replace(".", "-", regex=False)
    d = pd.to_datetime(c["Date added"], errors="coerce")
    s = pd.Series(d.to_numpy(), index=pd.Index(t, name="ticker"), name="add_date")
    s = s[~s.index.duplicated(keep="first")]
    return s.astype("datetime64[ns]")


@lru_cache(maxsize=4)
def _cached_load(snapshot_dir: str, universe_file: str, exclude: tuple[str, ...] = ()) -> DataContext:
    return DataContext._read(snapshot_dir, universe_file, exclude)
