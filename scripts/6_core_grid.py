"""
6_core_grid.py  -  STUDY 01: every pre-registered configuration, on the BUILD periods only.

  E1: 270 rules x pieces {1,2,3,4} x deadline {none,30,60,next} x fill {touch, 0.05% through} = 8,640
  E2: 255 rules x pieces {1,2,3,4} x deadline {none,30,60,next}                                = 4,080
  + reference lines: Buy & Hold, monthly DCA (1st trading day), perfect-timing ceiling (hindsight)

Every configuration is scored against Buy & Hold on the SAME period:
  d_xirr_bh  = strategy XIRR - Buy & Hold XIRR   (pts/yr, before selling)  <- the main number
  d_xirr_dca = strategy XIRR - monthly DCA XIRR
  ceiling_share = how much of the perfect-timing edge it captured

This script only RUNS and SAVES. Picking finalists (plateau rule, §9) is the next script.
Holdout stays locked.

Outputs: results/01_core_grid/runs.parquet (+ .csv), appended to results/master.csv
Run from project root:  python scripts/6_core_grid.py            (all cores)
                        python scripts/6_core_grid.py --quick    (1 in 20 configs, to test)
"""
import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.benchmarks import BuyHold, MonthlyDCA, PerfectTimingCeiling  # noqa: E402
from src.costs import IBKR_TIERED                                    # noqa: E402
from src.data import load                                             # noqa: E402
from src.engine import run                                            # noqa: E402
from src.metrics import summarize                                     # noqa: E402
from src.signals import default_grid                                  # noqa: E402
from src.strategy import DEADLINES, DipStrategy                       # noqa: E402

OUT = ROOT / "results" / "01_core_grid"
PERIOD = {"E1": "E1_build", "E2": "E2_build"}
PIECES = (1, 2, 3, 4)
FILLS = (0.0, 0.0005)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

_DATA = {}


def _init_worker():
    sys.path.insert(0, str(ROOT))
    for style, period in PERIOD.items():
        _DATA[style] = load(period)


def _run_one(job):
    style, rule, pieces, deadline, tt = job
    px, dist, start, end = _DATA[style]
    pol = DipStrategy(rule, style, pieces, deadline, tt)
    m = summarize(run(px, dist, start, end, pol, cost=IBKR_TIERED), IBKR_TIERED)
    m.update({"style": style, "rule": rule.name, "ref": rule.ref, "kind": rule.thr_kind,
              "thr": rule.thr, "reentry": rule.reentry if rule.reentry != "C" else f"C{rule.cooldown}",
              "pieces": pieces, "deadline": str(deadline), "trade_through_bp": tt * 1e4})
    return m


def jobs(quick=False):
    out = []
    for style in ("E1", "E2"):
        rules = [r for r in default_grid() if not (style == "E2" and r.ref == "open")]
        fills = FILLS if style == "E1" else (0.0,)
        out += [(style, r, p, d, tt) for r, p, d, tt in itertools.product(rules, PIECES, DEADLINES, fills)]
    return out[::20] if quick else out


def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------- reference lines
    _init_worker()
    refs = {}
    for style, period in PERIOD.items():
        px, dist, start, end = _DATA[style]
        for pol in (BuyHold(), MonthlyDCA(1), PerfectTimingCeiling()):
            refs[(style, pol.name)] = summarize(run(px, dist, start, end, pol), IBKR_TIERED)

    # ---------------- the grid
    todo = jobs(args.quick)
    print(f"Running {len(todo):,} configurations on {args.workers} cores "
          f"(E1: {sum(j[0]=='E1' for j in todo):,}, E2: {sum(j[0]=='E2' for j in todo):,})")
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        for k, m in enumerate(ex.map(_run_one, todo, chunksize=16), 1):
            rows.append(m)
            if k % 500 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"  {k:,}/{len(todo):,} done  ({el:.0f}s, ~{el/k*(len(todo)-k):.0f}s left)")
    df = pd.DataFrame(rows)

    for style in ("E1", "E2"):
        bh = refs[(style, "buy_hold")]["xirr_before_sell"]
        dca = refs[(style, "dca_monthly_d01")]["xirr_before_sell"]
        ceil = refs[(style, "ceiling_perfect_timing")]["xirr_before_sell"]
        s = df["style"] == style
        df.loc[s, "d_xirr_bh"] = df.loc[s, "xirr_before_sell"] - bh
        df.loc[s, "d_xirr_dca"] = df.loc[s, "xirr_before_sell"] - dca
        df.loc[s, "ceiling_share"] = (df.loc[s, "xirr_before_sell"] - bh) / (ceil - bh)

    df["study"] = "01_core_grid"
    df.to_parquet(OUT / "runs.parquet", index=False)
    df.to_csv(OUT / "runs.csv", index=False)
    refdf = pd.DataFrame([{**v, "style": k[0]} for k, v in refs.items()])
    refdf.to_csv(OUT / "reference_lines.csv", index=False)
    master = ROOT / "results" / "master.csv"
    old = pd.read_csv(master) if master.exists() else pd.DataFrame()
    if len(old):
        old = old[old["study"] != "01_core_grid"]
    pd.concat([old, df], ignore_index=True).to_csv(master, index=False)

    # ---------------- first look (description only; selection is the next script)
    for style in ("E1", "E2"):
        d = df[df["style"] == style]
        r = {k[1]: v for k, v in refs.items() if k[0] == style}
        bh, ce = r["buy_hold"]["xirr_before_sell"], r["ceiling_perfect_timing"]["xirr_before_sell"]
        section(f"{style}  ({PERIOD[style]}: {r['buy_hold']['start']} -> {r['buy_hold']['end']})")
        print(f"Buy & Hold {bh:.2%} | monthly DCA {r['dca_monthly_d01']['xirr_before_sell']:.2%} | "
              f"PERFECT-TIMING CEILING {ce:.2%} (hindsight, max possible edge {100*(ce-bh):+.2f} pts/yr)")
        dx = d["d_xirr_bh"] * 100
        print(f"\n{len(d):,} configurations, strategy XIRR minus Buy & Hold (pts/yr):")
        print(f"  worst {dx.min():+.2f} | 25% {dx.quantile(.25):+.2f} | median {dx.median():+.2f} | "
              f"75% {dx.quantile(.75):+.2f} | best {dx.max():+.2f}")
        print(f"  configurations beating Buy & Hold: {(dx > 0).sum():,} of {len(d):,} "
              f"({(dx > 0).mean():.1%})")
        print("\nmedian d_xirr_bh (pts/yr) by setting:")
        for col in ["pieces", "deadline", "reentry", "ref", "kind"] + (["trade_through_bp"] if style == "E1" else []):
            g = d.groupby(col)["d_xirr_bh"].median().mul(100).round(2)
            print(f"  {col:17s} " + "  ".join(f"{k}: {v:+.2f}" for k, v in g.items()))
        w = d["avg_wait_days"]
        print(f"\naverage days a euro waited in cash: median {w.median():.0f} (range {w.min():.0f}-{w.max():.0f})"
              f" | fees median {d['fees_pct_of_contrib'].median():.2%} of contributions")
    section("WROTE")
    print(f"  {OUT.relative_to(ROOT)}/runs.parquet + runs.csv ({len(df):,} runs), reference_lines.csv")
    print(f"  total time {time.time()-t0:.0f}s")
    print("DONE. Paste everything back.")


if __name__ == "__main__":
    main()
