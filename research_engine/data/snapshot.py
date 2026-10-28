"""Build the frozen market-data snapshot for the research-screening engine.

One run, one folder: data/snapshot_<date>/ with wide Parquet panels
(date x ticker) for open, high, low, close, adj_close, volume, dividends,
splits; SPY in the same panels; Fama-French daily factors and momentum;
manifest.json with sha256 per file; data_quality.json with per-ticker checks.

Never refetch into an existing snapshot folder.
"""
from __future__ import annotations

import hashlib, io, json, sys, time, zipfile, urllib.request, warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import yfinance as yf  # noqa: E402

START, END = "2012-01-01", "2026-01-01"
FIELDS = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
          "Adj Close": "adj_close", "Volume": "volume",
          "Dividends": "dividends", "Stock Splits": "splits"}
FRENCH = {
    "ff3_daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
    "mom_daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
}


def universe_tickers(root: Path) -> list[str]:
    names = sorted(p.name[:-len(" Hist.csv")] for p in (root / "legacy" / "Hist").glob("* Hist.csv"))
    return [n.replace(".", "-") for n in names]


def download_batch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    raw = yf.download(tickers, start=START, end=END, auto_adjust=False, actions=True,
                      group_by="ticker", threads=True, progress=False)
    out = {}
    for t in tickers:
        try:
            df = raw[t] if len(tickers) > 1 else raw
        except KeyError:
            continue
        df = df.dropna(how="all")
        if len(df):
            out[t] = df
    return out


def fetch_all(tickers: list[str], batch: int = 50, log=print) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        for attempt in range(3):
            try:
                got = download_batch(chunk)
                break
            except Exception as e:  # network hiccup
                log(f"batch {i//batch} attempt {attempt+1} failed: {e}")
                time.sleep(5 * (attempt + 1))
                got = {}
        frames.update(got)
        log(f"batch {i//batch + 1}/{(len(tickers)+batch-1)//batch}: {len(got)}/{len(chunk)} tickers")
    missing = [t for t in tickers if t not in frames]
    for t in missing:  # retry singly
        try:
            got = download_batch([t])
            frames.update(got)
        except Exception as e:
            log(f"single retry {t} failed: {e}")
    return frames


def to_panels(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    panels = {}
    for src, dst in FIELDS.items():
        cols = {}
        for t, df in frames.items():
            if src in df.columns:
                s = df[src].copy()
                s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
                cols[t] = s[~s.index.duplicated(keep="last")]
        panel = pd.DataFrame(cols).sort_index()
        panel.index.name = "date"
        panels[dst] = panel
    return panels


def fetch_french(dst: Path, log=print) -> dict[str, str]:
    out = {}
    for name, url in FRENCH.items():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            blob = r.read()
        z = zipfile.ZipFile(io.BytesIO(blob))
        csv_name = z.namelist()[0]
        text = z.read(csv_name).decode("latin-1")
        lines = text.splitlines()
        # find the header line: first line that starts with a comma (date column blank)
        start = next(i for i, l in enumerate(lines) if l.startswith(","))
        rows = []
        for l in lines[start + 1:]:
            if not l.strip() or not l.split(",")[0].strip().isdigit():
                break
            rows.append(l)
        hdr = [h.strip() for h in lines[start].split(",")]
        hdr[0] = "date"
        df = pd.read_csv(io.StringIO("\n".join([",".join(hdr)] + rows)))
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
        df = df.set_index("date").sort_index() / 100.0  # percent to decimal
        df = df.loc[START:END]
        path = dst / f"{name}.parquet"
        df.to_parquet(path)
        out[name] = url
        log(f"{name}: {len(df)} rows {df.index.min().date()} -> {df.index.max().date()}")
    return out


def quality(panels: dict[str, pd.DataFrame], root: Path) -> dict:
    close, adj = panels["close"], panels["adj_close"]
    splits, divs = panels["splits"], panels["dividends"]
    cal = close.index
    bdays = pd.bdate_range(cal.min(), cal.max())
    # NYSE holidays are the business days with no prices for >80% of tickers
    coverage = close.notna().mean(axis=1)
    report = {"n_tickers": int(close.shape[1]), "first_date": str(cal.min().date()),
              "last_date": str(cal.max().date()), "n_days": int(len(cal)),
              "n_business_days_span": int(len(bdays)),
              "low_coverage_days": [str(d.date()) for d in cal[coverage < 0.8]],
              "tickers": {}}
    # compare against repo adjusted closes on the overlap (2012..2024-04)
    repo_corr = {}
    for t in close.columns:
        s_close, s_adj = close[t].dropna(), adj[t].dropna()
        rec = {"first": str(s_close.index.min().date()) if len(s_close) else None,
               "last": str(s_close.index.max().date()) if len(s_close) else None,
               "n": int(len(s_close)),
               "nonpositive": int((s_close <= 0).sum() + (s_adj <= 0).sum()),
               "gaps_vs_calendar": int(len(cal[(cal >= s_close.index.min()) & (cal <= s_close.index.max())]) - len(s_close)) if len(s_close) else None}
        # split-date agreement: raw and adjusted daily returns must match except on split/dividend days
        r_raw, r_adj = s_close.pct_change(), s_adj.reindex(s_close.index).pct_change()
        ev = (splits[t].reindex(s_close.index).fillna(0) > 0) | (divs[t].reindex(s_close.index).fillna(0) > 0)
        diff = (r_raw - r_adj).abs()
        rec["raw_adj_mismatch_days"] = int((diff[~ev] > 1e-6).sum())
        rec["n_splits"] = int((splits[t].fillna(0) > 0).sum())
        rec["n_dividends"] = int((divs[t].fillna(0) > 0).sum())
        # overlap with repo file
        rp = root / "legacy" / "Hist" / f"{t.replace('-', '.')} Hist.csv"
        if rp.exists():
            old = pd.read_csv(rp, parse_dates=["Date"]).set_index("Date").iloc[:, 0]
            common = old.index.intersection(s_adj.index)
            if len(common) > 500:
                c = np.corrcoef(old.loc[common].pct_change().dropna(),
                                s_adj.loc[common].pct_change().dropna())[0, 1]
                rec["repo_return_corr"] = float(round(c, 6))
        report["tickers"][t] = rec
    report["tickers_missing_from_yahoo"] = []
    return report


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(root: Path, stamp: str | None = None) -> Path:
    stamp = stamp or date.today().isoformat()
    dst = root / "data" / f"snapshot_{stamp}"
    if dst.exists():
        sys.exit(f"{dst} exists; never refetch into an existing snapshot")
    dst.mkdir(parents=True)
    log_path = dst / "download.log"

    def log(msg: str):
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    tickers = universe_tickers(root)
    log(f"universe: {len(tickers)} tickers from Hist/; range {START} -> {END}")
    frames = fetch_all(tickers + ["SPY"], log=log)
    missing = [t for t in tickers if t not in frames]
    log(f"downloaded {len(frames)} series; missing: {missing}")
    panels = to_panels(frames)
    for name, panel in panels.items():
        panel.to_parquet(dst / f"{name}.parquet")
        log(f"{name}: {panel.shape}")
    french = fetch_french(dst, log=log)
    q = quality({k: v.drop(columns=["SPY"], errors="ignore") for k, v in panels.items()}, root)
    q["tickers_missing_from_yahoo"] = missing
    q["spy_rows"] = int(panels["close"]["SPY"].notna().sum()) if "SPY" in panels["close"] else 0
    (dst / "data_quality.json").write_text(json.dumps(q, indent=1))
    manifest = {
        "snapshot": stamp, "created": pd.Timestamp.now().isoformat(),
        "source": {"prices": "Yahoo Finance via yfinance " + yf.__version__, **french},
        "range": [START, END],
        "universe": "434 names from Hist/*.csv (current S&P 500 members with history from 2010); SURVIVOR UNIVERSE, no delisted names",
        "tickers": tickers, "missing": missing,
        "files": {p.name: sha256(p) for p in sorted(dst.glob("*.parquet"))},
        "fields": list(FIELDS.values()),
        "adjustment": "adj_close = Yahoo dividend+split adjusted as of snapshot date; close/high/low/open = raw unadjusted",
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log("manifest written")
    return dst


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    main(root)
