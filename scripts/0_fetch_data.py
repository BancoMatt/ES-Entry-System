"""
0_fetch_data.py  -  downloads all external daily data into data/raw/.
Runs on YOUR machine (the cloud sandbox is blocked from these sites).

Sources:
  fx/eurusd_ecb.csv        ECB official EURUSD reference rate (USD per 1 EUR), daily
  fx/eurusd_fred.csv       FRED DEXUSEU (NY noon rate) - backup / cross-check
  macro/dgs10.csv          FRED 10y US Treasury yield, daily (%)
  macro/ecb_dfr.csv        ECB deposit facility rate (%) - idle-cash interest variant
  etf/IUSA.AS.csv          real IUSA price + dividends (validation of reconstruction)
  etf/GSPC.csv             S&P 500 price index, daily
  etf/SP500TR.csv          S&P 500 total return index, daily (dividend factor)
  fx/EURUSD_yahoo.csv      Yahoo EURUSD=X - second backup
  macro/TNX_yahoo.csv      Yahoo ^TNX 10y yield - backup for FRED DGS10
MANUAL (script only checks it exists):
  macro/ie_data.xls        Shiller data, from shillerdata.com

Rules: never overwrites an existing file unless --force. Every file gets a
manifest entry (source, download time, rows, date range, sha256) so we always
know exactly which data vintage a result was built on.

Usage (from project root):
    python scripts/0_fetch_data.py                      # fetch only missing files
    python scripts/0_fetch_data.py --force --only IUSA.AS,GSPC,SP500TR,eurusd_ecb,SXR8.DE
          (--only = refetch just these; matches on file name, skips slow FRED)
"""
import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "manifest.json"
START = "1999-01-01"   # euro starts 1999 -> gives dot-com + GFC bears (history extension, 2026-09-24)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_manifest():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def record(man, path: Path, source: str, df: pd.DataFrame, date_col: str):
    d = pd.to_datetime(df[date_col])
    man[str(path.relative_to(ROOT)).replace("\\", "/")] = {
        "source": source,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "first": str(d.min().date()),
        "last": str(d.max().date()),
        "sha256": sha256(path),
    }
    print(f"  OK   {path.relative_to(ROOT)}  rows={len(df)}  {d.min().date()} -> {d.max().date()}")


def get_csv(url, tries=3, timeout=120):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except Exception as e:          # FRED is often slow: retry
            last = e
            print(f"    retry {i+1}/{tries} ({type(e).__name__})")
    raise last


# ---------------- individual sources ----------------
def ecb_series(key, value_name):
    url = (f"https://data-api.ecb.europa.eu/service/data/{key}"
           f"?format=csvdata&startPeriod={START}")
    df = get_csv(url)[["TIME_PERIOD", "OBS_VALUE"]]
    df.columns = ["date", value_name]
    return df.dropna(), url


def fred_series(sid, value_name):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={START}"
    df = get_csv(url)
    df.columns = ["date", value_name]
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")   # FRED uses '.' for missing
    return df.dropna(), url


def yahoo(ticker):
    import yfinance as yf
    h = yf.Ticker(ticker).history(start=START, auto_adjust=False, actions=True)
    if h.empty:
        raise RuntimeError("empty response")
    h = h.reset_index()
    h["Date"] = pd.to_datetime(h["Date"]).dt.tz_localize(None).dt.date
    return h.rename(columns={"Date": "date"}), f"yfinance:{ticker}"


JOBS = [
    ("fx/eurusd_ecb.csv",   lambda: ecb_series("EXR/D.USD.EUR.SP00.A", "usd_per_eur")),
    ("fx/eurusd_fred.csv",  lambda: fred_series("DEXUSEU", "usd_per_eur")),
    ("macro/dgs10.csv",     lambda: fred_series("DGS10", "yield_10y_pct")),
    ("macro/ecb_dfr.csv",   lambda: ecb_series("FM/B.U2.EUR.4F.KR.DFR.LEV", "dfr_pct")),
    ("etf/IUSA.AS.csv",     lambda: yahoo("IUSA.AS")),
    ("etf/GSPC.csv",        lambda: yahoo("^GSPC")),
    ("etf/SP500TR.csv",     lambda: yahoo("^SP500TR")),
    ("fx/EURUSD_yahoo.csv", lambda: yahoo("EURUSD=X")),
    ("macro/TNX_yahoo.csv", lambda: yahoo("^TNX")),   # 10y yield backup if FRED fails
    ("etf/SXR8.DE.csv",     lambda: yahoo("SXR8.DE")),  # iShares Core S&P 500 ACC, EUR, Xetra
]
OPTIONAL = {"fx/eurusd_fred.csv"}   # nice-to-have cross-check, not required


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download and overwrite")
    ap.add_argument("--only", default="", help="comma list of names to (re)fetch, e.g. IUSA.AS,SXR8.DE")
    args = ap.parse_args()
    only = [x.strip() for x in args.only.split(",") if x.strip()]

    man = load_manifest()
    failed = []
    print(f"Project root: {ROOT}\n")
    for rel, fn in JOBS:
        path = RAW / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if only and not any(o in rel for o in only):
            continue
        if path.exists() and not args.force:
            print(f"  skip {rel} (exists, use --force to refresh)")
            continue
        try:
            df, src = fn()
            df.to_csv(path, index=False)
            record(man, path, src, df, "date")
        except Exception as e:
            tag = "WARN (optional)" if rel in OPTIONAL else "FAIL"
            print(f"  {tag} {rel}: {type(e).__name__}: {e}")
            if rel not in OPTIONAL:
                failed.append(rel)

    # manual file
    shiller = RAW / "macro" / "ie_data.xls"
    if shiller.exists():
        print(f"  OK   macro/ie_data.xls present ({shiller.stat().st_size/1e3:.0f} KB)")
        man["data/raw/macro/ie_data.xls"] = {
            "source": "shillerdata.com (manual)",
            "registered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": sha256(shiller)}
    else:
        print("  TODO macro/ie_data.xls missing -> download manually from shillerdata.com")
        failed.append("macro/ie_data.xls")

    MANIFEST.write_text(json.dumps(man, indent=2))

    # quick sanity print
    print("\n--- sanity ---")
    for rel in ["fx/eurusd_ecb.csv", "etf/IUSA.AS.csv", "etf/GSPC.csv", "etf/SXR8.DE.csv"]:
        p = RAW / rel
        if p.exists():
            df = pd.read_csv(p)
            print(f"{rel}: first row\n{df.head(1).to_string(index=False)}\nlast row\n"
                  f"{df.tail(1).to_string(index=False)}\n")
    if (RAW / "etf/IUSA.AS.csv").exists():
        iu = pd.read_csv(RAW / "etf/IUSA.AS.csv")
        div = iu[iu.get("Dividends", 0) > 0]
        print(f"IUSA.AS dividends found: {len(div)} (tells us Dist vs Acc)")
        print(div[["date", "Dividends"]].tail(8).to_string(index=False))

    print("\nFAILED:", failed if failed else "none")
    print("Paste this whole output back.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
