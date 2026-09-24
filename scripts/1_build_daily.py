"""
1_build_daily.py  -  raw data -> clean daily panels in data/processed/.
READS raw files only. Writes ONLY to data/processed/ (safe to delete + rebuild).

Outputs
  data/processed/es_globex_daily.parquet    one row per ES Globex session
      session_date = date of the session's close. 18:00 ET opens the NEXT day's session.
  data/processed/es_euronext_daily.parquet  ES bars inside 09:00-17:30 Europe/Amsterdam
  data/processed/panel_daily.parquet        master panel on the IUSA (Euronext) trading calendar:
      iusa_* (real EUR prices, dividends), es_eu_*, es_gx_*, usd_per_eur (ECB), gspc_close

Rules enforced here
  - Unfinished sessions (e.g. today's) are DROPPED, never treated as a full day.
  - Nothing is forward-looking: each row only holds data from that session.
    (ECB FX is fixed ~14:15 CET, so it's for valuation, NOT for 09:00 signals.)
  - All thresholds printed below are DESCRIPTIVE (how often dips happen),
    not performance. Nothing is selected here.

Run from project root:  python scripts/1_build_daily.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

NY, AMS = "America/New_York", "Europe/Amsterdam"
DIPS = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02]
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# ---------------------------------------------------------------- ES 1-min
def load_es():
    files = sorted((RAW / "es_1min").glob("*.parquet"))
    cols = ["ts_event", "instrument_id", "open", "high", "low", "close", "volume"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    df = df.rename(columns={"ts_event": "ts"}).sort_values("ts").reset_index(drop=True)
    dups = df["ts"].duplicated().sum()
    if dups:
        raise SystemExit(f"{dups} duplicate timestamps across files - stop and check.")
    print(f"ES: {len(files)} files, {len(df):,} bars, {df.ts.min()} -> {df.ts.max()}")
    return df


def aggregate(df, key):
    """OHLCV per session + time of the low + contract info."""
    g = df.groupby(key, sort=True)
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                close=("close", "last"), volume=("volume", "sum"), n_bars=("ts", "size"),
                first_ts=("ts", "first"), last_ts=("ts", "last"),
                inst_first=("instrument_id", "first"), inst_last=("instrument_id", "last"))
    out["low_ts"] = df.loc[g["low"].idxmin().values, "ts"].values
    out["dip_depth"] = out["low"] / out["open"] - 1          # <= 0; most negative move from open
    out["roll_inside"] = out["inst_first"] != out["inst_last"]
    out.index.name = "date"
    out.index = pd.to_datetime(out.index)
    return out


def build_globex(df):
    ny = df["ts"].dt.tz_convert(NY)
    key = (ny + pd.Timedelta(hours=6)).dt.tz_localize(None).dt.normalize()
    gx = aggregate(df, key)
    # completeness: last session must reach the close (16:59 ET, or 17:14 ET pre-2015)
    last_ny = gx["last_ts"].dt.tz_convert(NY)
    close_min = np.where(gx.index.year < 2015, 17 * 60 + 10, 16 * 60 + 55)
    reached = (last_ny.dt.hour * 60 + last_ny.dt.minute) >= close_min
    final = gx.index == gx.index.max()
    dropped = gx[final & ~reached]
    gx = gx[~(final & ~reached)]
    gx = gx[gx.index.dayofweek < 5]                       # no weekend "sessions"
    gx["short"] = gx["n_bars"] < 600                      # holidays / early closes (kept, flagged)
    return gx, dropped


def build_euronext(df):
    ams = df["ts"].dt.tz_convert(AMS)
    mins = ams.dt.hour * 60 + ams.dt.minute
    m = (mins >= 9 * 60) & (mins < 17 * 60 + 30)          # bars starting 09:00 .. 17:29
    sub = df[m].copy()
    key = ams[m].dt.tz_localize(None).dt.normalize()
    eu = aggregate(sub, key)
    last_ams = eu["last_ts"].dt.tz_convert(AMS)
    reached = (last_ams.dt.hour * 60 + last_ams.dt.minute) >= 17 * 60 + 25
    final = eu.index == eu.index.max()
    dropped = eu[final & ~reached]
    eu = eu[~(final & ~reached)]
    eu = eu[eu.index.dayofweek < 5]
    eu["short"] = eu["n_bars"] < 400                      # full window = 510 bars
    return eu, dropped


# ---------------------------------------------------------------- daily files
def load_yahoo(name):
    d = pd.read_csv(RAW / "etf" / name, parse_dates=["date"]).set_index("date").sort_index()
    return d


def load_fx():
    fx = pd.read_csv(RAW / "fx" / "eurusd_ecb.csv", parse_dates=["date"]).set_index("date")
    return fx["usd_per_eur"].sort_index()


# ---------------------------------------------------------------- inventory
def inventory():
    section("INVENTORY of manual files (so I can read Shiller + iShares formats)")
    for sub in ["etf", "macro"]:
        for f in sorted((RAW / sub).iterdir()):
            if f.suffix.lower() == ".csv" and f.name in {
                    "IUSA.AS.csv", "GSPC.csv", "SP500TR.csv"}:
                continue
            print(f"\n--- {sub}/{f.name}  ({f.stat().st_size/1e3:.0f} KB)")
            try:
                if f.suffix.lower() in {".xls", ".xlsx"}:
                    xl = pd.ExcelFile(f)
                    print("sheets:", xl.sheet_names)
                    for sh in xl.sheet_names[:4]:
                        print(f"[{sh}] first 12 rows:")
                        print(xl.parse(sh, header=None, nrows=12).to_string())
                elif f.suffix.lower() == ".csv":
                    print(pd.read_csv(f, nrows=5).to_string())
                else:
                    print(f.read_bytes()[:400])
            except Exception as e:
                print(f"  pandas could not parse ({type(e).__name__}: {e})")
                print("  first bytes:", f.read_bytes()[:400])


# ---------------------------------------------------------------- main
def main():
    es = load_es()

    section("ES GLOBEX SESSIONS")
    gx, gx_drop = build_globex(es)
    print(f"sessions: {len(gx)}  {gx.index.min().date()} -> {gx.index.max().date()}")
    print("dropped unfinished:", [str(d.date()) for d in gx_drop.index] or "none")
    print("short (holiday/early close):", int(gx["short"].sum()),
          "| roll inside a session:", int(gx["roll_inside"].sum()), "(should be 0)")
    print(gx.groupby(gx.index.year).size().to_string())

    section("ES INSIDE EURONEXT WINDOW 09:00-17:30 Amsterdam")
    eu, eu_drop = build_euronext(es)
    print(f"days: {len(eu)}  {eu.index.min().date()} -> {eu.index.max().date()}")
    print("dropped unfinished:", [str(d.date()) for d in eu_drop.index] or "none")
    print("short:", int(eu["short"].sum()), "| roll inside window:", int(eu["roll_inside"].sum()),
          "(should be 0) | median bars/day:", int(eu["n_bars"].median()))

    del es

    # ---- IUSA + other daily
    section("IUSA.AS (real EUR prices)")
    iu = load_yahoo("IUSA.AS.csv")
    iu = iu.rename(columns=str.lower)[["open", "high", "low", "close", "volume", "dividends"]]
    iu.columns = ["iusa_" + c for c in iu.columns]
    iu = iu[iu.index.dayofweek < 5]
    prev = iu["iusa_close"].shift()
    flags = pd.DataFrame({
        "vol0": iu["iusa_volume"] == 0,
        "flat_bar": (iu["iusa_high"] == iu["iusa_low"]),
        "open_eq_prevclose": iu["iusa_open"] == prev,
        "bad_ohlc": (iu["iusa_low"] > iu[["iusa_open", "iusa_close"]].min(axis=1)) |
                    (iu["iusa_high"] < iu[["iusa_open", "iusa_close"]].max(axis=1)),
    })
    q = flags.groupby(iu.index.year).mean().mul(100).round(1)
    q.insert(0, "days", iu.groupby(iu.index.year).size())
    print("data-quality % per year (high flat/stale % = unreliable daily open/low):")
    print(q.to_string())
    r = iu["iusa_close"].pct_change()
    print("\nlargest daily moves:\n", r.abs().nlargest(8).to_string())
    a = iu.loc[iu.index <= "2026-09-22", "iusa_close"].iloc[-1]
    print(f"\nanchor check: IUSA close on/before 2026-09-22 = {a:.3f} (you quoted 67.58)")
    divs = iu.loc[iu["iusa_dividends"] > 0, "iusa_dividends"]
    print(f"Yahoo dividends: {len(divs)}, first {divs.index.min().date()} "
          f"last {divs.index.max().date()} (expect ~4/yr)")
    print(divs.groupby(divs.index.year).size().to_string())

    fx = load_fx()
    spx = load_yahoo("GSPC.csv")["Close"].rename("gspc_close")
    sptr = load_yahoo("SP500TR.csv")["Close"].rename("sptr_close")

    # ---- master panel on IUSA calendar
    panel = iu.copy()
    panel = panel.join(eu.add_prefix("es_eu_")[[f"es_eu_{c}" for c in
                        ["open", "high", "low", "close", "dip_depth", "low_ts", "n_bars", "short"]]])
    panel = panel.join(gx.add_prefix("es_gx_")[[f"es_gx_{c}" for c in
                        ["open", "high", "low", "close", "dip_depth", "low_ts", "n_bars", "short"]]])
    panel["usd_per_eur"] = fx.reindex(panel.index, method="ffill")
    panel["gspc_close"] = spx.reindex(panel.index, method="ffill")
    panel["sptr_close"] = sptr.reindex(panel.index, method="ffill")
    panel["iusa_dip_depth"] = panel["iusa_low"] / panel["iusa_open"] - 1

    section("PANEL COVERAGE")
    have_es = panel["es_eu_open"].notna()
    print(f"IUSA days: {len(panel)} | with ES Euronext window: {have_es.sum()} "
          f"| ES range starts {eu.index.min().date()}")
    miss = panel.loc[(panel.index >= eu.index.min()) & (panel.index <= eu.index.max()) & ~have_es]
    print(f"IUSA days inside ES range but missing ES bars: {len(miss)}",
          [str(d.date()) for d in miss.index[:15]])

    section("VALIDATION: does ES (Euronext window) track IUSA? (2010-06 onward)")
    v = panel[have_es & panel["iusa_open"].notna()].copy()
    v = v[(v.index >= "2010-06-07")]
    wk = v[["iusa_close", "gspc_close", "usd_per_eur", "es_eu_close"]].resample("W-FRI").last()
    wr = wk.pct_change().dropna()
    spx_eur = (1 + wr["gspc_close"]) / (1 + wr["usd_per_eur"]) - 1
    es_eur = (1 + wr["es_eu_close"]) / (1 + wr["usd_per_eur"]) - 1
    print(f"weekly corr IUSA vs S&P-in-EUR: {wr['iusa_close'].corr(spx_eur):.3f}")
    print(f"weekly corr IUSA vs ES(17:30)-in-EUR: {wr['iusa_close'].corr(es_eur):.3f}")
    print(f"intraday dip_depth corr (IUSA own vs ES window): "
          f"{v['iusa_dip_depth'].corr(v['es_eu_dip_depth']):.3f}")
    for x in [0.01, 0.0125]:
        a_ = v["iusa_dip_depth"] <= -x
        b_ = v["es_eu_dip_depth"] <= -x
        print(f"  dip {x:.2%}: ES says yes {b_.sum()} | IUSA says yes {a_.sum()} | both {(a_ & b_).sum()}")

    section("DESCRIPTIVE: % of days where low <= open*(1-x)  (frequency only, no selection)")
    rows = {}
    for name, s in [("ES globex", gx["dip_depth"]), ("ES euronext", eu["dip_depth"]),
                    ("IUSA own", panel["iusa_dip_depth"].dropna())]:
        rows[name] = {f"{x:.2%}": round((s <= -x).mean() * 100, 1) for x in DIPS}
    print(pd.DataFrame(rows).T.to_string())
    yr = pd.DataFrame({f"{x:.2%}": (eu["dip_depth"] <= -x).groupby(eu.index.year).sum()
                       for x in DIPS})
    print("\nES euronext window, number of days per year:\n", yr.to_string())

    # ---- write
    gx.to_parquet(PROC / "es_globex_daily.parquet")
    eu.to_parquet(PROC / "es_euronext_daily.parquet")
    panel.to_parquet(PROC / "panel_daily.parquet")
    section("WROTE")
    for f in ["es_globex_daily", "es_euronext_daily", "panel_daily"]:
        p = PROC / f"{f}.parquet"
        print(f"  {p.relative_to(ROOT)}  {p.stat().st_size/1e6:.2f} MB")

    inventory()
    print("\nDONE. Paste everything back.")


if __name__ == "__main__":
    main()
