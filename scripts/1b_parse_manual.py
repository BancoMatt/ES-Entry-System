"""
1b_parse_manual.py  -  parses the two manually downloaded files. READ ONLY on raw.

1) iShares fund file (data/raw/etf/iShares-*.xls)
   It's not a real .xls: it's "SpreadsheetML" (Excel 2003 XML), which pandas can't read.
   We parse the XML ourselves, dump EVERY sheet to data/processed/ishares/<sheet>.csv,
   and print each sheet's name + first rows so we can see the distributions / NAV tables.

2) Shiller ie_data.xls (data/raw/macro/)
   -> data/processed/shiller_monthly.parquet
   TRAP handled: Shiller dates are floats, so October is 2020.1 (NOT 2020.10 = Jan).
   Month is taken as round(frac * 100). No lag applied here; the lag is applied
   at signal time (Shiller interpolates quarterly earnings across months, so the
   monthly E is NOT known in real time - we'll use a fixed, conservative lag).

Run from project root:  python scripts/1b_parse_manual.py
"""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SS = "{urn:schemas-microsoft-com:office:spreadsheet}"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 40)


def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# ------------------------------------------------------------------ iShares XML
def parse_spreadsheetml(path: Path) -> dict:
    raw = path.read_bytes()
    raw = raw.replace(b"\xef\xbb\xbf", b"")                 # file has a double BOM
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        print(f"  strict XML parse failed ({e}); trying lxml recover mode")
        from lxml import etree                              # pip install lxml
        root = etree.fromstring(raw, parser=etree.XMLParser(recover=True, huge_tree=True))
    sheets = {}
    for ws in root.iter(f"{SS}Worksheet"):
        name = ws.get(f"{SS}Name")
        rows = []
        for row in ws.iter(f"{SS}Row"):
            vals, col = [], 0
            for cell in row.findall(f"{SS}Cell"):
                idx = cell.get(f"{SS}Index")                # skipped empty cells
                if idx:
                    while col < int(idx) - 1:
                        vals.append(None)
                        col += 1
                d = cell.find(f"{SS}Data")
                vals.append(d.text if d is not None else None)
                col += 1
            rows.append(vals)
        width = max((len(r) for r in rows), default=0)
        sheets[name] = pd.DataFrame([r + [None] * (width - len(r)) for r in rows])
    return sheets


def do_ishares():
    files = sorted((RAW / "etf").glob("iShares*"))
    section("iSHARES FUND FILE")
    if not files:
        print("no iShares file in data/raw/etf/")
        return
    f = files[0]
    print("file:", f.name)
    sheets = parse_spreadsheetml(f)
    out = PROC / "ishares"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in sheets.items():
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name or "sheet")
        df.to_csv(out / f"{safe}.csv", index=False, header=False)
        print(f"\n--- sheet '{name}'  shape={df.shape}  -> data/processed/ishares/{safe}.csv")
        print(df.head(12).to_string())
        if len(df) > 12:
            print("  ...last 3 rows:")
            print(df.tail(3).to_string())


# ------------------------------------------------------------------ Shiller
SHILLER_COLS = {0: "date_raw", 1: "P", 2: "D", 3: "E", 4: "CPI", 5: "date_frac",
                6: "GS10", 7: "real_P", 8: "real_D", 9: "real_TR_P", 10: "real_E",
                11: "real_TR_scaled_E", 12: "CAPE", 14: "TR_CAPE", 16: "excess_CAPE_yield"}


def do_shiller():
    section("SHILLER ie_data.xls")
    f = RAW / "macro" / "ie_data.xls"
    raw = pd.read_excel(f, sheet_name="Data", header=None)
    hdr = raw.iloc[:8].fillna("").astype(str).agg(" ".join).str.strip()
    print("header check (col -> text):")
    for i in SHILLER_COLS:
        print(f"  {i:2d} {SHILLER_COLS[i]:18s} <- '{re.sub(' +', ' ', hdr.iloc[i])}'")

    df = raw.iloc[8:, list(SHILLER_COLS)].copy()
    df.columns = list(SHILLER_COLS.values())
    df["date_raw"] = pd.to_numeric(df["date_raw"], errors="coerce")
    df = df.dropna(subset=["date_raw"])
    yr = np.floor(df["date_raw"]).astype(int)
    mo = np.round((df["date_raw"] - yr) * 100).astype(int)
    bad = ~mo.between(1, 12)
    if bad.any():
        raise SystemExit(f"bad month parse on rows: {df.loc[bad, 'date_raw'].head().tolist()}")
    df.index = pd.to_datetime(dict(year=yr, month=mo, day=1))
    df.index.name = "month"
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop(columns=["date_raw"])

    # sanity: months strictly consecutive, October parsed right
    gaps = df.index.to_series().diff().dt.days.dropna()
    print(f"\nrows {len(df)} | {df.index.min():%Y-%m} -> {df.index.max():%Y-%m} "
          f"| non-monthly steps: {int((~gaps.between(28, 31)).sum())}")
    print("October check (should show month 10):", df.loc["2020-10"].index[0].strftime("%Y-%m"))
    print("\nlast 8 months (E / D often blank or estimated at the end):")
    print(df[["P", "D", "E", "GS10", "CAPE", "excess_CAPE_yield"]].tail(8).to_string())
    last_e = df["E"].last_valid_index()
    last_cape = df["CAPE"].last_valid_index()
    print(f"\nlast month with E: {last_e:%Y-%m} | with CAPE: {last_cape:%Y-%m}")
    print("\n2010-2026 sample, every January:")
    s = df.loc["2010":]
    print(s[s.index.month == 1][["P", "E", "D", "GS10", "CAPE"]].to_string())
    df.to_parquet(PROC / "shiller_monthly.parquet")
    print("\nWROTE data/processed/shiller_monthly.parquet")


if __name__ == "__main__":
    PROC.mkdir(parents=True, exist_ok=True)
    do_ishares()
    do_shiller()
    print("\nDONE. Paste everything back.")
