"""
src/data.py  -  the ONLY way the engine gets data. Enforces the holdout lock.

Periods are fixed by the pre-registration (prereg-v1):
  E1_build  2009-01-02 -> 2018-12-31   (intraday style, real IUSA bars only)
  E2_build  1999-01-04 -> 2018-12-31   (end-of-day style, closes only)
  holdout   2019-01-01 -> last day     LOCKED: needs holdout=True, every unlock is logged

History BEFORE a period's start is always loaded too (so 50-day highs, volatility etc.
are ready on day one). Nothing AFTER the period's end is ever loaded.
"""
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PERIODS = {
    "E1_build": ("2009-01-02", "2018-12-31"),
    "E2_build": ("1999-01-04", "2018-12-31"),
    "holdout": ("2019-01-01", None),
}


def _log_unlock(period):
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip() or "no-git"
    except Exception:
        commit = "no-git"
    line = (f"- {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC | period={period} | "
            f"commit={commit} | command: {' '.join(sys.argv)}\n")
    log = ROOT / "notes" / "holdout_log.md"
    if not log.exists():
        log.write_text("# Holdout log\n\nEvery time the 2019+ data is opened, a line is added here.\n\n")
    with open(log, "a") as f:
        f.write(line)


def load(period: str, holdout: bool = False):
    """Returns (prices up to period end, distributions up to period end, start, end)."""
    if period not in PERIODS:
        raise ValueError(f"unknown period {period}")
    if period == "holdout" and not holdout:
        raise PermissionError("Holdout is locked. Pass holdout=True (--holdout) deliberately. It gets logged.")
    start, end = PERIODS[period]
    px = pd.read_parquet(PROC / "pit_prices.parquet")
    dist = pd.read_parquet(PROC / "pit_distributions.parquet")
    end = pd.Timestamp(end) if end else px.index.max()
    px = px[px.index <= end]
    dist = dist[dist["ex_date"] <= end]
    if period == "holdout":
        _log_unlock(period)
    return px, dist, pd.Timestamp(start), end
