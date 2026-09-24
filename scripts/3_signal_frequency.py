"""
3_signal_frequency.py  -  HOW OFTEN does each entry rule fire?  (no profits, no returns)

Why: fees make tiny orders expensive, so the budget (EUR 500/quarter) supports roughly
2-3 buys per quarter = ~8-12 per year. We choose drop sizes by how often they fire,
NOT by how much money they would have made. That keeps us clear of curve-fitting.

HOLDOUT LOCK: this script only ever loads data up to 2018-12-31.

Also: PROXY CHECK. Before 2009 we only have the real fund's daily CLOSE; the day's
open/high/low come from the S&P's own daily shape (US hours). Here we rebuild that
same proxy for 2009-2018, where REAL IUSA prices exist, and compare signal counts.
If they agree, pre-2009 results can be trusted for that rule; if not, we know by how much.

Outputs: results/03_signal_frequency/{per_year.csv, summary.csv}
Run from project root:  python scripts/3_signal_frequency.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pit import known_at_open_features       # noqa: E402
from src.signals import Rule, entries, default_grid  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "03_signal_frequency"
BUILD_END = pd.Timestamp("2018-12-31")
TARGET = (8, 12)          # buys per year that fit the fee constraint (2-3 per quarter)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)


def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def fmt_thr(r):
    return f"{r.thr:.2%}" if r.thr_kind == "fixed" else f"{r.thr:g}x"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px = pd.read_parquet(PROC / "pit_prices.parquet")
    px = px[px.index <= BUILD_END]                      # <-- holdout lock
    feat = known_at_open_features(px)
    print(f"BUILD period only: {px.index.min().date()} -> {px.index.max().date()} "
          f"({len(px)} days). Eras: {px['era'].value_counts().to_dict()}")

    # ------------------------------------------------------------ all rules
    rows, per_year = [], {}
    for r in default_grid():
        e = entries(px, feat, r)
        valid = e["n"].notna()
        yr = e.loc[valid, "n"].groupby(e.index[valid].year).sum()
        full_years = yr[yr.index.map(lambda y: valid[e.index.year == y].mean() > 0.9)]
        per_year[r.name] = yr
        rows.append({
            "rule": r.name, "ref": r.ref, "kind": r.thr_kind, "thr": fmt_thr(r),
            "reentry": r.reentry if r.reentry != "C" else f"C{r.cooldown}",
            "years": len(full_years),
            "mean_per_yr": full_years.mean(), "min_yr": full_years.min(),
            "max_yr": full_years.max(), "zero_years": int((full_years == 0).sum()),
        })
    summ = pd.DataFrame(rows)
    pd.DataFrame(per_year).to_csv(OUT / "per_year.csv")
    summ.to_csv(OUT / "summary.csv", index=False)

    # ------------------------------------------------------------ readable tables
    def pivot(kind, value):
        s = summ[summ["kind"] == kind]
        order = list(dict.fromkeys(s["thr"]))
        t = s.pivot_table(index=["ref", "reentry"], columns="thr", values=value, aggfunc="first")
        return t[order].round(1)

    section("AVERAGE BUYS PER YEAR - fixed % drop   (target band 8-12 = 2-3 per quarter)")
    print(pivot("fixed", "mean_per_yr").to_string())
    section("WORST YEAR (fewest buys) - fixed % drop   (0 = a year with no buys at all)")
    print(pivot("fixed", "min_yr").to_string())
    section("AVERAGE BUYS PER YEAR - volatility-scaled drop (k x 20-day vol)")
    print(pivot("vol", "mean_per_yr").to_string())
    section("WORST YEAR - volatility-scaled drop")
    print(pivot("vol", "min_yr").to_string())

    band = summ[(summ["mean_per_yr"].between(*TARGET)) & (summ["min_yr"] >= 2)]
    section(f"RULES IN THE TARGET BAND: {TARGET[0]}-{TARGET[1]} buys/yr on average AND >= 2 in the worst year")
    print(f"{len(band)} of {len(summ)} rules")
    print(band.sort_values(["ref", "reentry", "kind"]).to_string(index=False))

    # ------------------------------------------------------------ year-by-year example
    section("YEAR BY YEAR, a few examples (shows calm vs crisis years)")
    ex = ["prev_close|1.25%|-", "high_20d|3.00%|A", "high_20d|2.00%|B", "high_20d|2xvol|A"]
    ex = [e for e in ex if e in per_year]
    print(pd.DataFrame({k: per_year[k] for k in ex}).fillna(0).astype(int).to_string())

    # ------------------------------------------------------------ proxy validation
    section("PROXY CHECK on 2009-2018: real IUSA bars vs S&P-shaped proxy bars")
    real = px[px["era"] == "real"].copy()
    spx = pd.read_csv(ROOT / "data" / "raw" / "etf" / "GSPC.csv", parse_dates=["date"]).set_index("date")
    spx.columns = [c.lower() for c in spx.columns]
    ratio = spx[["open", "high", "low"]].div(spx["close"], axis=0).reindex(real.index)
    prox = real.copy()
    for col in ["open", "high", "low"]:
        prox[col] = real["close"] * ratio[col]
    prox = prox.dropna(subset=["open"])
    real = real.loc[prox.index]
    fr, fp = known_at_open_features(real), known_at_open_features(prox)
    checks = [Rule("prev_close", "fixed", 0.0125), Rule("prev_close", "fixed", 0.02),
              Rule("open", "fixed", 0.0125), Rule("high_20d", "fixed", 0.03, "A"),
              Rule("high_20d", "fixed", 0.02, "B"), Rule("high_20d", "vol", 2.0, "A")]
    out = []
    for r in checks:
        a = entries(real, fr, r)["n"].fillna(0) > 0
        b = entries(prox, fp, r)["n"].fillna(0) > 0
        yrs = len(np.unique(real.index.year))
        out.append({"rule": r.name, "real_per_yr": a.sum() / yrs, "proxy_per_yr": b.sum() / yrs,
                    "same_day_%": 100 * (a & b).sum() / max(1, (a | b).sum())})
    print(pd.DataFrame(out).round(1).to_string(index=False))
    print("\nHow to read: close-based rules (prev_close, high_Nd) should match well because the")
    print("proxy's CLOSE is the real fund close; 'open' rules can differ a lot (US-hours open).")
    print(f"\nWROTE {OUT.relative_to(ROOT)}/summary.csv and per_year.csv")
    print("DONE. Paste everything back.")


if __name__ == "__main__":
    main()
