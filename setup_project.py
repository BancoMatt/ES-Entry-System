"""
setup_project.py  -  run ONCE to create the ES_ENTRY_SYSTEM project skeleton.

What it does:
  1. Creates the folder tree (never deletes or overwrites anything).
  2. Writes requirements.txt, .gitignore, config.yaml, README.md (only if missing).
  3. Optionally COPIES the ES 1-min parquets from your old project (read-only on
     the source), then verifies every copy byte-for-byte with SHA-256.

Usage:
    python setup_project.py
    python setup_project.py --es-src "~/FINANCE/VWAP_CVA strategy/data/raw/es_1min"
"""
import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

FOLDERS = [
    "data/raw/es_1min",      # ES 1-min parquets. READ ONLY after copy.
    "data/raw/fx",           # EURUSD (ECB + Yahoo backup)
    "data/raw/macro",        # Shiller ie_data.xls, 10y yield, ECB rates
    "data/raw/etf",          # IUSA.AS price + dividends, ^GSPC
    "data/processed",        # everything derived. Safe to delete and rebuild.
    "scripts",               # numbered pipeline scripts
    "results",               # one subfolder per run
    "figures",
    "notes",                 # pre-registration, decisions log
]

REQUIREMENTS = """pandas>=2.1
pyarrow>=14
numpy>=1.26
scipy>=1.11
matplotlib>=3.8
requests>=2.31
yfinance>=0.2.40
openpyxl>=3.1
xlrd>=2.0
pyyaml>=6.0
"""

GITIGNORE = """.venv/
__pycache__/
*.pyc
data/
results/
figures/
.ipynb_checkpoints/
"""

CONFIG = """# ES Entry System - FIXED parameters. Change only with a dated note in notes/decisions.md
project:
  start: "2010-06-07"          # first ES bar available (Databento GLBX history starts here)
  build_end: "2018-12-31"      # in-sample: build + select
  holdout_start: "2019-01-01"  # out-of-sample: touched ONCE by finalists

capital:
  contribution_eur: 500
  frequency: quarterly         # first trading day of each quarter
  cash_floor_eur: 0

instrument:
  name: "iShares Core S&P 500 UCITS ETF (IUSA, Euronext Amsterdam)"
  anchor_price_eur: 67.58
  anchor_date: "2026-09-22"

sessions:                       # both are tested
  globex: {tz: "America/New_York", start: "18:00", end: "17:00"}   # prior-day 18:00 -> 17:00
  euronext: {tz: "Europe/Amsterdam", start: "09:00", end: "17:30"}

costs:                          # fill in real values from broker
  fee_per_order_eur: null
  spread_bps: null
  dividend_tax: 0.26
  cash_interest: 0.0            # variant: ECB deposit rate

paths:
  es_1min: "data/raw/es_1min"
"""

README = """# ES Entry System

Does a rules-based entry system (dip + trend + index valuation + sizing) beat
Buy & Hold and monthly DCA on the same EUR 500/quarter, buying IUSA?

Rules: no leverage, no look-ahead, no parameter fitting, raw data read-only.
Pipeline: scripts/0_fetch_data.py -> 1_build_daily.py -> 2_engine.py ->
3_strategies.py -> 4_metrics.py -> 5_report.py -> 6_robustness.py
"""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_if_missing(path: Path, text: str):
    if path.exists():
        print(f"  keep   {path.relative_to(ROOT)} (exists)")
    else:
        path.write_text(text, encoding="utf-8")
        print(f"  create {path.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--es-src", default=None, help="folder with es_1min_YYYY.parquet to copy")
    args = ap.parse_args()

    print("Project root:", ROOT)
    for f in FOLDERS:
        (ROOT / f).mkdir(parents=True, exist_ok=True)
        print(f"  dir    {f}")

    write_if_missing(ROOT / "requirements.txt", REQUIREMENTS)
    write_if_missing(ROOT / ".gitignore", GITIGNORE)
    write_if_missing(ROOT / "config.yaml", CONFIG)
    write_if_missing(ROOT / "README.md", README)
    write_if_missing(ROOT / "notes" / "decisions.md",
                     "# Decisions log\n\n- 2026-09-23: project created, config.yaml v1.\n")

    if args.es_src:
        src = Path(args.es_src).expanduser()
        dst = ROOT / "data/raw/es_1min"
        files = sorted(src.glob("*.parquet"))
        print(f"\nCopying {len(files)} parquet files from {src}")
        if not files:
            print("  NONE FOUND - check the path.")
        for f in files:
            t = dst / f.name
            if t.exists():
                print(f"  skip   {f.name} (already there)")
                continue
            shutil.copy2(f, t)
            ok = sha256(f) == sha256(t)
            print(f"  copy   {f.name:28s} {f.stat().st_size/1e6:7.1f} MB  sha256 match={ok}")
            if not ok:
                raise SystemExit(f"CHECKSUM MISMATCH on {f.name}. Stop.")

    print("\nDone. Next: create the venv and install requirements (see instructions).")


if __name__ == "__main__":
    main()
