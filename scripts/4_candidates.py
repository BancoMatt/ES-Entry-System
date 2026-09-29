"""
4_candidates.py  -  builds the FROZEN candidate lists required by the pre-registration.

A rule is a candidate if, on its own build period and execution style, it fires
8-12 times per year on average AND at least 2 times in its worst full year.
Only signal COUNTS are used. No prices-to-profit, no returns.

  E1 (intraday limit): real IUSA data only, 2009-01-02 -> 2018-12-31
  E2 (end of day):     closes only, 1999-01-04 -> 2018-12-31

Writes (these small CSVs ARE committed to git, they're part of the pre-registration):
  notes/candidates_E1.csv, notes/candidates_E2.csv
  notes/frequency_E1_all.csv, notes/frequency_E2_all.csv   (every rule, for the heatmaps)

Run from project root:  python scripts/4_candidates.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pit import known_at_open_features         # noqa: E402
from src.signals import default_grid, frequency    # noqa: E402

PROC, NOTES = ROOT / "data" / "processed", ROOT / "notes"
BUILD_END = pd.Timestamp("2018-12-31")
BAND, MIN_WORST = (8, 12), 2
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)


def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def main():
    px_all = pd.read_parquet(PROC / "pit_prices.parquet")
    px_all = px_all[px_all.index <= BUILD_END]                 # holdout lock
    setups = {
        "E1": px_all[px_all["era"] == "real"],
        "E2": px_all,
    }
    for style, px in setups.items():
        feat = known_at_open_features(px)
        rules = [r for r in default_grid() if not (style == "E2" and r.ref == "open")]
        summ, _ = frequency(px, feat, rules, style)
        summ["candidate"] = summ["mean_per_yr"].between(*BAND) & (summ["min_yr"] >= MIN_WORST)
        summ.round(3).to_csv(NOTES / f"frequency_{style}_all.csv", index=False)
        cand = summ[summ["candidate"]].drop(columns="candidate")
        cand.round(3).to_csv(NOTES / f"candidates_{style}.csv", index=False)

        section(f"{style}: {px.index.min().date()} -> {px.index.max().date()}  "
                f"| {len(rules)} rules tested | {len(cand)} candidates")
        show = cand.copy()
        show["thr"] = [f"{t:.2%}" if k == "fixed" else f"{t:g}x vol"
                       for t, k in zip(show["thr"], show["kind"])]
        print(show[["ref", "reentry", "kind", "thr", "mean_per_yr", "min_yr", "max_yr"]]
              .sort_values(["ref", "reentry", "kind"]).round(1).to_string(index=False))
        print("\ncandidates per family (ref x re-entry):")
        print(cand.groupby(["ref", "reentry"]).size().unstack(fill_value=0).to_string())

    section("NEXT")
    print("Commit notes/ (preregistration.md + these CSVs) and tag prereg-v1 BEFORE the engine runs.")
    print("DONE. Paste everything back.")


if __name__ == "__main__":
    main()
