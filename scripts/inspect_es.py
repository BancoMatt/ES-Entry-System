"""
inspect_es.py  -  READ ONLY. Never writes, never modifies the raw data.

Prints everything we need to know about the ES 1-min parquet files before
building the pipeline: schema, timestamp/timezone, session coverage, price
scale, contract/symbol columns, roll gaps, duplicates, missing data.

Usage (from project root):
    python inspect_es.py
    python inspect_es.py --dir "path/to/es_1min"

Paste the FULL output back.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "raw", "es_1min")   # <project>/data/raw/es_1min

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def find_col(cols, candidates):
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def get_ts(df):
    """Return a tz-aware UTC timestamp Series (or naive if tz unknown) + source description."""
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, index=range(len(df))), f"index '{df.index.name}' tz={df.index.tz}"
    tcol = find_col(df.columns, ["ts_event", "timestamp", "datetime", "date_time", "time", "date", "ts"])
    if tcol is None:
        return None, "NO TIMESTAMP FOUND"
    s = df[tcol]
    if not pd.api.types.is_datetime64_any_dtype(s):
        if pd.api.types.is_integer_dtype(s):
            unit = "ns" if s.iloc[0] > 1e17 else ("ms" if s.iloc[0] > 1e11 else "s")
            s = pd.to_datetime(s, unit=unit, utc=True)
        else:
            s = pd.to_datetime(s, errors="coerce")
    return s.reset_index(drop=True), f"column '{tcol}' dtype={df[tcol].dtype}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.parquet")))
    section("FILES")
    print("dir:", args.dir)
    if not files:
        print("NO PARQUET FILES FOUND. Pass --dir with the correct folder.")
        sys.exit(1)
    for f in files:
        print(f"  {os.path.basename(f):35s} {os.path.getsize(f)/1e6:8.1f} MB")

    # ---------- detailed look at first + last file ----------
    for f in [files[0], files[-1]]:
        section(f"DETAIL: {os.path.basename(f)}")
        df = pd.read_parquet(f)
        print("shape:", df.shape)
        print("index type:", type(df.index).__name__, "| index name:", df.index.name)
        print("\ndtypes:\n", df.dtypes.to_string())
        print("\nhead:\n", df.head(5).to_string())
        print("\ntail:\n", df.tail(5).to_string())
        print("\nnulls per column:\n", df.isna().sum().to_string())

    # ---------- per-file scan ----------
    section("PER-FILE SCAN (all years)")
    all_daily = []
    sym_changes = []
    for f in files:
        df = pd.read_parquet(f)
        ts, ts_src = get_ts(df)
        if ts is None:
            print(os.path.basename(f), "-> no timestamp found. Columns:", list(df.columns))
            continue
        o = find_col(df.columns, ["open", "o"])
        h = find_col(df.columns, ["high", "h"])
        l = find_col(df.columns, ["low", "l"])
        c = find_col(df.columns, ["close", "c", "last"])
        v = find_col(df.columns, ["volume", "vol", "v"])
        sym = find_col(df.columns, ["instrument_id", "contract", "raw_symbol", "symbol", "ticker"])

        d = pd.DataFrame({"ts": ts})
        for name, col in [("open", o), ("high", h), ("low", l), ("close", c), ("volume", v)]:
            d[name] = df[col].values if col else np.nan
        if sym:
            d["sym"] = df[sym].astype(str).values

        tz = getattr(d["ts"].dt, "tz", None)
        dup = d["ts"].duplicated().sum()
        step = d["ts"].diff().dt.total_seconds().median()
        bad_ohlc = ((d["high"] < d[["open", "close"]].max(axis=1)) |
                    (d["low"] > d[["open", "close"]].min(axis=1))).sum()
        print(f"{os.path.basename(f):28s} rows={len(d):8d} {d['ts'].min()} -> {d['ts'].max()} "
              f"tz={tz} dupTS={dup} medStep={step}s badOHLC={bad_ohlc} "
              f"close[min/max]={d['close'].min():.2f}/{d['close'].max():.2f}")
        if f == files[0]:
            print("   timestamp source:", ts_src, "| symbol col:", sym)

        # symbol / contract changes (roll detection)
        if sym:
            chg = d["sym"] != d["sym"].shift()
            idx = d.index[chg][1:]
            for i in idx:
                sym_changes.append((d.at[i, "ts"], d.at[i - 1, "sym"], d.at[i, "sym"],
                                    d.at[i - 1, "close"], d.at[i, "open"]))

        # hour-of-day coverage in UTC and New York
        tsu = d["ts"] if tz is not None else d["ts"].dt.tz_localize("UTC")
        d["hour_utc"] = tsu.dt.tz_convert("UTC").dt.hour
        d["date_ny"] = tsu.dt.tz_convert("America/New_York").dt.date
        all_daily.append(d[["ts", "hour_utc", "date_ny", "open", "high", "low", "close", "volume"]])
        del df

    big = pd.concat(all_daily, ignore_index=True)

    section("HOUR-OF-DAY COVERAGE (UTC, % of bars)  -> tells us RTH-only vs 24h Globex")
    print((big["hour_utc"].value_counts(normalize=True).sort_index() * 100).round(2).to_string())

    section("CONTRACT / SYMBOL CHANGES (first 40)")
    if sym_changes:
        for t, a, b, pc, no in sym_changes[:40]:
            print(f"  {t}  {a} -> {b}  last_close={pc}  next_open={no}  gap={no - pc:+.2f}")
        print(f"  total changes: {len(sym_changes)}")
    else:
        print("  no symbol column -> checking for roll-like gaps instead (below)")

    section("LARGEST BAR-TO-BAR GAPS (close -> next open), top 30  -> roll jumps / data holes")
    big = big.sort_values("ts").reset_index(drop=True)
    big["gap"] = big["open"] - big["close"].shift()
    big["gap_pct"] = big["gap"] / big["close"].shift() * 100
    big["dt_min"] = big["ts"].diff().dt.total_seconds() / 60
    top = big.reindex(big["gap_pct"].abs().sort_values(ascending=False).index).head(30)
    print(top[["ts", "close", "open", "gap", "gap_pct", "dt_min"]].to_string())

    section("MISSING TRADING DAYS CHECK (NY dates with bars, per year)")
    days = pd.Series(pd.to_datetime(big["date_ny"].unique()))
    print(days.dt.year.value_counts().sort_index().to_string())

    section("PRICE SCALE SAMPLES (daily close, 1st trading day of each year)")
    first = big.groupby(pd.to_datetime(big["date_ny"]).dt.year).first()
    print(first[["ts", "close"]].to_string())

    print("\nDONE. Paste everything above back.")


if __name__ == "__main__":
    main()
