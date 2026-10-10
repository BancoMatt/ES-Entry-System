"""
8_funding_calendar.py  -  STUDY 06a (benchmark part of pre-registered study 06, funding)
                          + a CALENDAR CURIOSITY (descriptive only, added in notes/decisions.md)

PART A - funding. Same EUR 2,000/yr, three ways the money can ARRIVE, and what to do with it.
  Money arrives once a year (EUR 2,000, first trading day of January)
      lump sum on arrival ("annual B&H") | spread over 4 quarters | spread over 12 months
      | hindsight: the year's lowest close (ceiling, not a strategy)
  Money arrives every quarter (EUR 500)  <- the base case of the whole project
      B&H on arrival (base) | save it up, invest once a year in January | monthly DCA
      | hindsight: the quarter's lowest close
  Money arrives every month (EUR 166.67)
      B&H on arrival
  Each strategy is compared with "invest on arrival" for the SAME arrival schedule:
  that's the decision you actually control. The three "invest on arrival" lines are also
  compared with each other (does HOW OFTEN you get paid matter?).

PART B - which day of the year? (E2 build period 1999-2018, closes only)
  For every trading day k of the year (1 = first trading day of January ... ~250):
    "arrives on day k"  the EUR 2,000 lands on day k every year and is invested that day.
                        -> pure calendar effect: is some day of the year special?
    "waits until day k" the EUR 2,000 lands on day 1 and sits in cash until day k.
                        -> what does waiting for a chosen day cost?
  The waiting line is compared with the break-even formula from fig04.

PART C - Monte Carlo of the calendar: every path picks a RANDOM trading day in each year
  ("arrives on day k" version). 10,000 paths = the spread of pure timing luck.
  Bounded by hindsight: every year at its lowest close (best) / highest close (worst).

Holdout stays locked (build periods only). Nothing here can change the finalists.
Outputs: results/08_funding/*.csv, figures/fig08..fig10*.png, rows in results/master.csv
Run from project root:  python scripts/8_funding_calendar.py
                        MC_PATHS=500 python scripts/8_funding_calendar.py   (quick test)
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import plots                                                       # noqa: E402
from src.benchmarks import (BuyHold, InvestOnDays, MonthlyDCA,             # noqa: E402
                            PerfectTimingCeiling, SpreadBuys, extreme_day_of_year,
                            nth_day_of_year)
from src.costs import IBKR_TIERED                                           # noqa: E402
from src.data import load                                                   # noqa: E402
from src.engine import Engine                                               # noqa: E402
from src.metrics import summarize, xirr                                     # noqa: E402

OUT = ROOT / "results" / "08_funding"
FIG = ROOT / "figures"
COST = IBKR_TIERED
PER_YEAR = 2000.0
MC_PATHS = int(os.environ.get("MC_PATHS", 10_000))
SEED = 2026
WORKERS = max(1, (os.cpu_count() or 2) - 1)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

_DATA = {}


def _init_worker():
    sys.path.insert(0, str(ROOT))
    for p in ("E1_build", "E2_build"):
        _DATA[p] = load(p)


def run_one(period, policy, freq="Q", contrib_days=None):
    """One engine run. contrib_days = trading-day indices where EUR 2,000 arrives (overrides freq)."""
    px, dist, start, end = _DATA[period]
    eng = Engine(px, dist, start, end, policy, cost=COST, per_year=PER_YEAR, freq=freq)
    if contrib_days is not None:
        eng.contrib = {eng.dates[i]: PER_YEAR for i in contrib_days}
    res = eng.run()
    t = res.totals
    left = t["contributed"] + t["div_gross"] - t["div_tax"]
    right = t["cash_end"] + t["spent_on_shares"] + t["fees"]
    assert abs(left - right) < 1e-6, f"accounting broken in {policy.name}"
    return res


def _first_of_year(eng):
    return nth_day_of_year(eng.dates, 1)


# ---------------------------------------------------------------- PART A
FUNDING = [
    # (arrival, label, policy factory, contribution freq, is_reference)
    ("yearly", "lump sum on arrival (annual B&H)", lambda: BuyHold(), "A", True),
    ("yearly", "spread over 4 quarters", lambda: SpreadBuys("Q"), "A", False),
    ("yearly", "spread over 12 months", lambda: SpreadBuys("M"), "A", False),
    ("yearly", "hindsight: year's lowest close", lambda: PerfectTimingCeiling(), "A", False),
    ("quarterly", "B&H on arrival (base)", lambda: BuyHold(), "Q", True),
    ("quarterly", "save up, invest once a year (Jan)", lambda: InvestOnDays(_first_of_year, "save_up_annual"), "Q", False),
    ("quarterly", "monthly DCA", lambda: MonthlyDCA(1), "Q", False),
    ("quarterly", "hindsight: quarter's lowest close", lambda: PerfectTimingCeiling(), "Q", False),
    ("monthly", "B&H on arrival", lambda: BuyHold(), "M", True),
]


def _job_funding(job):
    period, k = job
    arrival, label, make, freq, is_ref = FUNDING[k]
    m = summarize(run_one(period, make(), freq), COST)
    m.update({"period": period, "arrival": arrival, "label": label, "is_reference": is_ref})
    return m


# ---------------------------------------------------------------- PART B + C
def _job_calendar(job):
    kind, k = job
    px, dist, start, end = _DATA["E2_build"]
    dates = px.index[(px.index >= start) & (px.index <= end)]
    days = nth_day_of_year(dates, k)
    if kind == "arrives":
        res = run_one("E2_build", BuyHold(), contrib_days=days)
    else:
        res = run_one("E2_build", InvestOnDays(days, f"wait_until_{k}"), freq="A")
    m = summarize(res, COST)
    m.update({"variant": kind, "k": k})
    return m


def _job_mc(days):
    res = run_one("E2_build", BuyHold(), contrib_days=days)
    t = res.totals
    return xirr(res.flows + [(t["end_date"], t["value_end"])])


def pmap(fn, jobs, label, chunksize=1):
    t0 = time.time()
    out = []
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker) as ex:
        for i, r in enumerate(ex.map(fn, jobs, chunksize=chunksize), 1):
            out.append(r)
            if i % max(1, len(jobs) // 5) == 0 or i == len(jobs):
                print(f"  {label}: {i:,}/{len(jobs):,}  ({time.time() - t0:.0f}s)")
    return out


def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def approx_date(dates, k):
    """Typical calendar date of the k-th trading day (median over the years)."""
    doy = [dates[i].dayofyear for i in nth_day_of_year(dates, k)]
    return (pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(np.median(doy)) - 1)).strftime("%d %b")


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _init_worker()

    # ============ PART A
    jobs = [(p, k) for p in ("E1_build", "E2_build") for k in range(len(FUNDING))]
    fund = pd.DataFrame(pmap(_job_funding, jobs, "funding runs"))
    for p in ("E1_build", "E2_build"):
        m = fund["period"] == p
        base = fund[m & (fund["label"] == "B&H on arrival (base)")]["xirr_before_sell"].iloc[0]
        fund.loc[m, "d_vs_quarterly_bh"] = fund.loc[m, "xirr_before_sell"] - base
        for arr in ("yearly", "quarterly", "monthly"):
            ma = m & (fund["arrival"] == arr)
            ref = fund[ma & fund["is_reference"]]["xirr_before_sell"].iloc[0]
            fund.loc[ma, "d_vs_own_arrival"] = fund.loc[ma, "xirr_before_sell"] - ref
    fund["study"] = "06_funding"
    fund.to_csv(OUT / "funding.csv", index=False)

    # ============ PART B
    px, dist, start, end = _DATA["E2_build"]
    dates = px.index[(px.index >= start) & (px.index <= end)]
    K = int(pd.Series(dates.year).value_counts().min())
    cal_jobs = [(kind, k) for kind in ("arrives", "waits") for k in range(1, K + 1)]
    scan = pd.DataFrame(pmap(_job_calendar, cal_jobs, "calendar scan", chunksize=8))
    scan["approx_date"] = scan["k"].map(lambda k: approx_date(dates, k))

    # break-even prediction for the waiting line (same formula as fig04, no extra fee)
    lump = scan[(scan["variant"] == "arrives") & (scan["k"] == 1)].iloc[0]
    mu = lump["xirr_before_sell"]
    day1 = nth_day_of_year(dates, 1)
    ybar = np.mean([(end - dates[i]).days / 365.25 for i in day1])
    w = {k: np.mean([(dates[b] - dates[a]).days for a, b in zip(day1, nth_day_of_year(dates, k))])
         for k in range(1, K + 1)}
    scan["wait_days"] = scan["k"].map(w)
    scan["pred_waits"] = mu - mu * scan["wait_days"] / 365 / ybar
    scan.to_csv(OUT / "calendar_scan.csv", index=False)

    # ============ PART C
    rng = np.random.default_rng(SEED)
    yr = pd.Series(range(len(dates)), index=dates)
    groups = [g.to_numpy() for _, g in yr.groupby(dates.year)]
    mc_days = [[int(rng.choice(g)) for g in groups] for _ in range(MC_PATHS)]
    mc = np.array(pmap(_job_mc, mc_days, "Monte Carlo", chunksize=64))
    pd.DataFrame({"path": range(MC_PATHS), "xirr_before_sell": mc}).to_csv(OUT / "calendar_mc.csv", index=False)
    hind = {}
    for how in ("min", "max"):
        hind[how] = _job_mc(extreme_day_of_year(dates, px.loc[dates, "close"].to_numpy(), how))

    master = ROOT / "results" / "master.csv"
    old = pd.read_csv(master, low_memory=False) if master.exists() else pd.DataFrame()
    if len(old):
        old = old[old["study"] != "06_funding"]
    pd.concat([old, fund], ignore_index=True).to_csv(master, index=False)

    # ============ PRINT
    for p in ("E1_build", "E2_build"):
        d = fund[fund["period"] == p]
        section(f"PART A  {p}: {d['start'].iloc[0]} -> {d['end'].iloc[0]}  (EUR 2,000/yr, IBKR Tiered, whole shares)")
        for arr, head in [("yearly", "money arrives ONCE A YEAR (EUR 2,000 in January)"),
                          ("quarterly", "money arrives EVERY QUARTER (EUR 500)  <- base case"),
                          ("monthly", "money arrives EVERY MONTH (EUR 166.67)")]:
            print(f"\n{head}")
            t = d[d["arrival"] == arr]
            for _, r in t.iterrows():
                print(f"  {r['label']:38s} XIRR {r['xirr_before_sell']:6.2%} (after sell {r['xirr_after_sell']:6.2%}) | "
                      f"vs invest-on-arrival {100 * r['d_vs_own_arrival']:+.2f} | vs quarterly B&H {100 * r['d_vs_quarterly_bh']:+.2f} | "
                      f"fees {r['fees_pct_of_contrib']:.2%} | wait {r['avg_wait_days']:5.0f}d | buys {r['n_buys']:4d} | "
                      f"final EUR {r['final_value']:,.0f}")

    section("PART B  which day of the year?  E2 build 1999-2018, EUR 2,000 once a year")
    a = scan[scan["variant"] == "arrives"].set_index("k")
    wv = scan[scan["variant"] == "waits"].set_index("k")
    best, worst = a["xirr_before_sell"].idxmax(), a["xirr_before_sell"].idxmin()
    print(f"trading days per year used: 1..{K}")
    print("\n'arrives on day k' (pure calendar effect):")
    print(f"  day 1 ({a.loc[1, 'approx_date']})  {a.loc[1, 'xirr_before_sell']:.2%}")
    print(f"  best  day {best} (~{a.loc[best, 'approx_date']})  {a.loc[best, 'xirr_before_sell']:.2%}")
    print(f"  worst day {worst} (~{a.loc[worst, 'approx_date']})  {a.loc[worst, 'xirr_before_sell']:.2%}")
    print(f"  median of all days {a['xirr_before_sell'].median():.2%} | spread best-worst "
          f"{100 * (a['xirr_before_sell'].max() - a['xirr_before_sell'].min()):.2f} pts/yr")
    print("  NOTE: the best of ~250 days is a hindsight pick, not a strategy.")
    print("\n'waits until day k' (money already there on day 1):")
    for k in [1, 21, 63, 126, 189, K]:
        print(f"  invest on day {k:3d} (~{wv.loc[k, 'approx_date']}, waited {wv.loc[k, 'wait_days']:5.0f}d): "
              f"{wv.loc[k, 'xirr_before_sell']:.2%}   formula predicts {wv.loc[k, 'pred_waits']:.2%}")
    print(f"  days that beat investing on day 1: {(wv['xirr_before_sell'] > wv.loc[1, 'xirr_before_sell']).sum()} of {K - 1}")
    print(f"  corr(actual, formula) {wv['xirr_before_sell'].corr(wv['pred_waits']):.2f}")

    section(f"PART C  Monte Carlo: random trading day every year, {MC_PATHS:,} paths")
    q = np.nanpercentile(mc, [5, 25, 50, 75, 95])
    print(f"  mean {np.nanmean(mc):.2%} | std {100 * np.nanstd(mc):.2f} pts")
    print(f"  5% {q[0]:.2%} | 25% {q[1]:.2%} | median {q[2]:.2%} | 75% {q[3]:.2%} | 95% {q[4]:.2%}")
    print(f"  paths beating 'day 1 every year' ({a.loc[1, 'xirr_before_sell']:.2%}): {np.mean(mc > a.loc[1, 'xirr_before_sell']):.0%}")
    print(f"  hindsight, every year at its LOWEST close:  {hind['min']:.2%}")
    print(f"  hindsight, every year at its HIGHEST close: {hind['max']:.2%}")

    charts(fund, scan, mc, hind, K, dates)
    section("WROTE")
    print(f"  {OUT.relative_to(ROOT)}/funding.csv, calendar_scan.csv, calendar_mc.csv")
    print("  figures/fig08_funding.png, fig09_calendar_scan.png, fig10_calendar_montecarlo.png")
    print("DONE. Paste everything back.")


# ---------------------------------------------------------------- charts
def charts(fund, scan, mc, hind, K, dates):
    import matplotlib.pyplot as plt
    plots.style()
    C = plots.SERIES
    better, worse = "#2a78d6", "#e34948"

    # fig08 funding: 2 periods x 3 questions, one metric per panel
    panels = [("If the money arrives ONCE A YEAR", "yearly", "d_vs_own_arrival", "vs lump sum on arrival"),
              ("If the money arrives EVERY QUARTER", "quarterly", "d_vs_own_arrival", "vs B&H on arrival"),
              ("Does HOW OFTEN you get paid matter?", None, "d_vs_quarterly_bh", "vs quarterly B&H")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.2))
    for row, p in enumerate(("E1_build", "E2_build")):
        d = fund[fund["period"] == p]
        for col, (title, arr, metric, xl) in enumerate(panels):
            ax = axes[row, col]
            if arr:
                t = d[(d["arrival"] == arr) & ~d["is_reference"]]
            else:
                t = d[d["is_reference"] & (d["arrival"] != "quarterly")].copy()
                t["label"] = t["arrival"].map({"yearly": "paid yearly, invest on arrival",
                                               "monthly": "paid monthly, invest on arrival"})
            v = (t[metric] * 100).to_numpy()
            labels = t["label"].tolist()
            cols = [C[2] if "hindsight" in lb else (better if x > 0 else worse) for lb, x in zip(labels, v)]
            y = np.arange(len(v))[::-1]
            ax.barh(y, v, color=cols, height=0.55)
            ax.axvline(0, color=plots.INK, linewidth=1)
            lim = max(0.3, np.abs(v).max() * 1.45)
            ax.set_xlim(-lim, lim)
            for yi, x in zip(y, v):
                ax.text(x + (0.02 * lim if x >= 0 else -0.02 * lim), yi, f"{x:+.2f}", va="center",
                        ha="left" if x >= 0 else "right", fontsize=8, color=plots.INK2)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel(f"XIRR {xl} (pts/yr)", fontsize=8)
            per = "2009-2018" if p == "E1_build" else "1999-2018"
            ax.set_title(f"{title}\n{per}", fontsize=9)
    fig.suptitle("Same EUR 2,000 a year: does it pay to wait, spread it out, or get paid less often?",
                 fontsize=11, weight="bold", color=plots.INK)
    fig.tight_layout()
    plots.note(fig, "Blue = better, red = worse, green = hindsight ceiling (not a strategy). IBKR Tiered fees, whole shares, "
                    "26% dividend tax, uninvested cash earns 0%. XIRR before selling.")
    plots.save(fig, FIG / "fig08_funding.png")

    # fig09 calendar scan
    a = scan[scan["variant"] == "arrives"].sort_values("k")
    w = scan[scan["variant"] == "waits"].sort_values("k")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(a["k"], a["xirr_before_sell"] * 100, color=C[0], linewidth=1.6)
    ax.plot(w["k"], w["xirr_before_sell"] * 100, color=C[1], linewidth=1.6)
    ax.plot(w["k"], w["pred_waits"] * 100, color=plots.INK2, linewidth=1, linestyle=(0, (4, 3)))
    xe = K + 3
    ax.text(xe, a["xirr_before_sell"].iloc[-1] * 100, "  arrives on day k,\n  invested that day", color=C[0], fontsize=8, va="center")
    ax.text(xe, w["xirr_before_sell"].iloc[-1] * 100, "  arrives in January,\n  waits until day k", color=C[1], fontsize=8, va="center")
    ax.text(xe, w["pred_waits"].iloc[-1] * 100, "  break-even formula\n  (waiting only)", color=plots.INK2, fontsize=8, va="center")
    ai = a.set_index("k")["xirr_before_sell"] * 100
    for k, lab in [(ai.idxmax(), "best day"), (ai.idxmin(), "worst day")]:
        ax.plot([k], [ai[k]], "o", color=C[0], markersize=5)
        right = k > 0.8 * K
        ax.annotate(f"{lab}: #{k} (~{a.set_index('k').loc[k, 'approx_date']})\n{ai[k]:.2f}%", (k, ai[k]),
                    textcoords="offset points", xytext=(-8 if right else 0, 6 if lab == "best day" else -26),
                    ha="right" if right else "center", fontsize=8, color=plots.INK2)
    lo = min(a["xirr_before_sell"].min(), w["xirr_before_sell"].min(), w["pred_waits"].min()) * 100
    hi = max(a["xirr_before_sell"].max(), w["xirr_before_sell"].max()) * 100
    ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.15 * (hi - lo))
    s = pd.Series(range(len(dates)), index=dates)
    ticks, labels = [], []
    for mth in range(1, 13):
        ks = [int((g.index.month < mth).sum()) + 1 for _, g in s.groupby(dates.year)]
        ticks.append(int(np.median(ks)))
        labels.append(pd.Timestamp(2001, mth, 1).strftime("%b"))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(1, K + 45)
    ax.spines["bottom"].set_bounds(1, K)
    ax.set_xlabel("trading day of the year when the EUR 2,000 is invested (month = where that day usually falls)")
    ax.set_ylabel("XIRR before selling (%/yr)")
    ax.set_title("1999-2018: does the day of the year matter? (one EUR 2,000 buy per year)")
    plots.note(fig, "Blue: pure calendar effect, every euro measured from the day it arrives. Orange: the money is already there "
                    "in January and waits in cash. The best/worst day are hindsight picks among ~250 tries.")
    plots.save(fig, FIG / "fig09_calendar_scan.png")

    # fig10 Monte Carlo
    fig, ax = plt.subplots(figsize=(12, 4.8))
    v = mc[~np.isnan(mc)] * 100
    ax.hist(v, bins=60, color=C[0], edgecolor=plots.SURFACE, linewidth=0.4)
    top = ax.get_ylim()[1]
    marks = [(ai[1], "day 1 every year\n(lump sum on arrival)", plots.INK),
             (ai.max(), "best fixed day\n(hindsight)", C[3]),
             (ai.min(), "worst fixed day\n(hindsight)", C[3]),
             (hind["min"] * 100, "every year at its\nLOWEST close", C[2]),
             (hind["max"] * 100, "every year at its\nHIGHEST close", worse)]
    for x, lab, col in marks:
        ax.axvline(x, color=col, linewidth=1.3)
    for j, (x, lab, col) in enumerate(sorted(marks)):
        ax.text(x, top * (1.02 + 0.14 * (j % 3)), lab, color=col, fontsize=7.5, ha="center", va="bottom",
                bbox=dict(facecolor=plots.SURFACE, edgecolor="none", pad=1.5))
    ax.set_ylim(0, top * 1.5)
    ax.set_xlabel("XIRR before selling (%/yr)")
    ax.set_ylabel("number of paths")
    p5, p95 = np.percentile(v, [5, 95])
    ax.set_title(f"1999-2018: {len(v):,} random 'one day per year' investors. 90% landed between {p5:.2f}% and {p95:.2f}%")
    plots.note(fig, "Each path invests EUR 2,000 once a year on a random trading day (money arrives that day). "
                    "Hindsight lines are the best and worst possible day in every single year: no rule can reach them.")
    plots.save(fig, FIG / "fig10_calendar_montecarlo.png")


if __name__ == "__main__":
    main()
