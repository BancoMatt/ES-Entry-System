"""
5_benchmarks.py  -  STUDY 02: Buy & Hold vs monthly DCA, on both build periods.

First real use of the engine. No dip strategy runs here. This answers:
  - What return does "just invest every EUR 500 on arrival" give?
  - What does monthly DCA give, and how much does the chosen DAY of the month matter?
  - How much do fees and dividend taxes cost in each case?

Holdout stays locked (build periods only).
Outputs: results/02_benchmarks/runs.csv  (+ appended to results/master.csv)
Run from project root:  python scripts/5_benchmarks.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.benchmarks import BuyHold, MonthlyDCA  # noqa: E402
from src.costs import IBKR_TIERED               # noqa: E402
from src.data import load                        # noqa: E402
from src.engine import run                       # noqa: E402
from src.metrics import summarize                # noqa: E402

OUT = ROOT / "results" / "02_benchmarks"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def pct(x):
    return f"{x:.2%}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for period in ["E2_build", "E1_build"]:
        px, dist, start, end = load(period)
        policies = [BuyHold()] + [MonthlyDCA(d) for d in range(1, 29)]
        for pol in policies:
            res = run(px, dist, start, end, pol, cost=IBKR_TIERED)
            m = summarize(res, IBKR_TIERED)
            # accounting check on every single run
            t = res.totals
            left = t["contributed"] + t["div_gross"] - t["div_tax"]
            right = t["cash_end"] + t["spent_on_shares"] + t["fees"]
            assert abs(left - right) < 1e-6, f"accounting broken in {pol.name}: {left} vs {right}"
            m.update({"study": "02_benchmarks", "period": period})
            rows.append(m)

        df = pd.DataFrame([r for r in rows if r["period"] == period])
        bh = df[df["policy"] == "buy_hold"].iloc[0]
        d1 = df[df["policy"] == "dca_monthly_d01"].iloc[0]
        dca = df[df["policy"].str.startswith("dca")]

        section(f"{period}: {bh['start']} -> {bh['end']}  (EUR 500/quarter, IBKR Tiered, whole shares)")
        show = pd.DataFrame({
            "Buy & Hold": bh, "Monthly DCA (1st trading day)": d1}).T[
            ["xirr_before_sell", "xirr_after_sell", "contributed", "final_value", "profit",
             "fees", "fees_pct_of_contrib", "div_tax", "avg_cash_pct", "max_drawdown",
             "n_buys", "avg_order_eur"]]
        fmt = show.copy()
        for c in ["xirr_before_sell", "xirr_after_sell", "fees_pct_of_contrib", "avg_cash_pct", "max_drawdown"]:
            fmt[c] = fmt[c].map(pct)
        for c in ["contributed", "final_value", "profit", "fees", "div_tax", "avg_order_eur"]:
            fmt[c] = fmt[c].map(lambda x: f"{x:,.0f}")
        print(fmt.T.to_string())

        diff = dca["xirr_before_sell"] - bh["xirr_before_sell"]
        print(f"\nMonthly DCA on EVERY day of the month (1..28), XIRR before selling:")
        print(f"  worst day   {dca['xirr_before_sell'].min():.2%}  "
              f"(day {int(dca.loc[dca['xirr_before_sell'].idxmin(), 'policy'][-2:])})")
        print(f"  median      {dca['xirr_before_sell'].median():.2%}")
        print(f"  best day    {dca['xirr_before_sell'].max():.2%}  "
              f"(day {int(dca.loc[dca['xirr_before_sell'].idxmax(), 'policy'][-2:])})")
        print(f"  vs Buy & Hold: DCA is {diff.median()*100:+.2f} pts/yr (median), "
              f"range {diff.min()*100:+.2f} to {diff.max()*100:+.2f}")
        print(f"  DCA days that beat Buy & Hold: {(diff > 0).sum()} of 28")

    allrows = pd.DataFrame(rows)
    allrows.to_csv(OUT / "runs.csv", index=False)
    master = ROOT / "results" / "master.csv"
    old = pd.read_csv(master) if master.exists() else pd.DataFrame()
    if len(old):
        old = old[old["study"] != "02_benchmarks"]            # re-running replaces, never duplicates
    pd.concat([old, allrows], ignore_index=True).to_csv(master, index=False)
    section("WROTE")
    print(f"  {OUT.relative_to(ROOT)}/runs.csv  ({len(allrows)} runs)  and results/master.csv")
    print("  accounting check passed on every run")
    print("DONE. Paste everything back.")


if __name__ == "__main__":
    main()
