"""
fetch_es_update.py  -  pulls the missing ES 1-min bars (2026-07-01 -> latest)
from Databento in EXACTLY the same format as the existing es_1min_YYYY files.

- Same dataset/schema/symbol as your original pull: GLBX.MDP3, ohlcv-1m, ES.c.0
- Shows the COST first and asks for confirmation before downloading.
- Writes a NEW file (es_1min_2026_h2.parquet). Never touches existing files.
- Checks the schema matches the existing 2026 file and the join is seamless.

Setup (fish shell):
    pip install databento
    set -x DATABENTO_API_KEY "db-XXXXXXXX"      # from databento.com -> API keys
Run from project root:
    python scripts/fetch_es_update.py
"""
from datetime import datetime, timezone
from pathlib import Path

import databento as db
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ES_DIR = ROOT / "data" / "raw" / "es_1min"
OUT = ES_DIR / "es_1min_2026_h2.parquet"
PREV = ES_DIR / "es_1min_2026.parquet"
START = "2026-07-01T00:00:00Z"


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT.name} already exists. Delete it yourself if you want to re-pull.")

    client = db.Historical()  # reads DATABENTO_API_KEY
    rng = client.metadata.get_dataset_range(dataset="GLBX.MDP3")
    end = rng["end"] if isinstance(rng, dict) else rng.end
    print("GLBX.MDP3 available until:", end)

    params = dict(dataset="GLBX.MDP3", schema="ohlcv-1m", symbols=["ES.c.0"],
                  stype_in="continuous", start=START, end=end)
    cost = client.metadata.get_cost(**params)
    size = client.metadata.get_billable_size(**params)
    print(f"Request: ES.c.0 ohlcv-1m {START} -> {end}")
    print(f"Cost: ${cost:.4f}   size: {size/1e6:.2f} MB")
    if input("Download? [y/N] ").strip().lower() != "y":
        raise SystemExit("Cancelled.")

    data = client.timeseries.get_range(**params)
    df = data.to_df().reset_index()          # ts_event back to a column, like the originals

    # ---- schema check against the existing 2026 file ----
    prev = pd.read_parquet(PREV)
    df = df[prev.columns]                    # same column order
    for c in prev.columns:
        if df[c].dtype != prev[c].dtype:
            try:
                df[c] = df[c].astype(prev[c].dtype)
            except Exception:
                print(f"  note: dtype {c}: new={df[c].dtype} old={prev[c].dtype}")
    print("\ncolumns match:", list(df.columns) == list(prev.columns))

    # ---- seam check ----
    last_old, first_new = prev["ts_event"].max(), df["ts_event"].min()
    overlap = (df["ts_event"] <= last_old).sum()
    print(f"old ends {last_old} | new starts {first_new} | overlapping bars: {overlap}")
    print(f"old last close {prev['close'].iloc[-1]} | new first open {df['open'].iloc[0]}")
    print(f"rows {len(df)} | {df['ts_event'].min()} -> {df['ts_event'].max()}")
    print("instrument_ids:", df["instrument_id"].unique().tolist())
    if overlap:
        raise SystemExit("Overlap found - not writing. Paste this output.")

    df.to_parquet(OUT, index=False)
    print(f"\nWROTE {OUT.relative_to(ROOT)}  pulled {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")


if __name__ == "__main__":
    main()
