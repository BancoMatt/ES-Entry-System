"""
9_holdout.py  -  pre-registration §10: the ONE look at 2019 -> last day.

Scope (fixed in notes/decisions.md BEFORE opening):
  benchmarks   quarterly B&H (base) | yearly B&H (EUR 2,000 every January) | save up, invest
               once a year | monthly DCA | hindsight: quarter's lowest close | hindsight: year's
               lowest close (yearly money)
  dip rules    the two NOT QUALIFIED reference configs from results/07_plateau/finalists.csv
               + the WHOLE grid (12,720 configs) as a DISTRIBUTION ONLY. Nothing is picked,
               ranked or promoted from it.
  checks       build result vs holdout result for every config (does the past predict the future?)
               break-even formula on the holdout (is it still all waiting + fees?)

Two modes:
  python scripts/9_holdout.py --dry-run    rehearsal on 2014-2018 (build data, nothing unlocked).
                                           Must pass first. Outputs go to results/09_holdout_DRYRUN/.
  python scripts/9_holdout.py --holdout    THE run. Pre-flight checks, then opens 2019+ once
                                           (logged in notes/holdout_log.md).
Outputs: results/09_holdout/*.csv, figures/fig11..fig14*.png, notes/holdout_results.md, master.csv
"""
import argparse
import itertools
import os
import subprocess
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
                            PerfectTimingCeiling, nth_day_of_year)
from src.costs import IBKR_TIERED                                           # noqa: E402
from src.data import load                                                   # noqa: E402
from src.engine import contribution_schedule, run                          # noqa: E402
from src.metrics import summarize                                           # noqa: E402
from src.signals import Rule, default_grid                                  # noqa: E402
from src.strategy import DEADLINES, DipStrategy                             # noqa: E402

COST = IBKR_TIERED
PIECES = (1, 2, 3, 4)
FILLS = (0.0, 0.0005)
PRIMARY_TT = {"E1": 5.0, "E2": 0.0}
WORKERS = max(1, (os.cpu_count() or 2) - 1)
DRY_START, DRY_END = "2014-01-01", "2018-12-31"
EVENTS = [("2020-02-19", "2020-03-23", "Covid crash"), ("2022-01-03", "2022-10-12", "2022 bear market")]
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

_D = {}


def _init(data):
    sys.path.insert(0, str(ROOT))
    _D["px"], _D["dist"], _D["start"], _D["end"] = data


def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# ---------------------------------------------------------------- policies
def _first_of_year(eng):
    return nth_day_of_year(eng.dates, 1)


BENCH = [
    # (key, label, factory, how the money arrives: Q = EUR 500 a quarter, A = EUR 2,000 each January)
    ("bh_q", "quarterly B&H (base)", lambda: BuyHold(), "Q"),
    ("bh_y", "yearly B&H (EUR 2,000 each January)", lambda: BuyHold(), "A"),
    ("save_up", "save up, invest once a year (Jan)", lambda: InvestOnDays(_first_of_year, "save_up_annual"), "Q"),
    ("dca", "monthly DCA", lambda: MonthlyDCA(1), "Q"),
    ("ceil_q", "hindsight: quarter's lowest close", lambda: PerfectTimingCeiling(), "Q"),
    ("ceil_y", "hindsight: year's lowest close (yearly money)", lambda: PerfectTimingCeiling(), "A"),
]


def rule_of(r):
    re_ = r["reentry"]
    if isinstance(re_, str) and re_.startswith("C"):
        return Rule(r["ref"], r["kind"], float(r["thr"]), "C", cooldown=int(re_[1:]))
    return Rule(r["ref"], r["kind"], float(r["thr"]), re_)


def deadline_of(d):
    return int(d) if str(d).isdigit() else str(d)


def ref_policy(r):
    return DipStrategy(rule_of(r), r["style"], int(r["pieces"]), deadline_of(r["deadline"]),
                       float(r["trade_through_bp"]) / 1e4)


def grid_jobs(quick=False):
    out = []
    for style in ("E1", "E2"):
        rules = [r for r in default_grid() if not (style == "E2" and r.ref == "open")]
        fills = FILLS if style == "E1" else (0.0,)
        out += [(style, r, p, d, tt) for r, p, d, tt in itertools.product(rules, PIECES, DEADLINES, fills)]
    return out[::19] if quick else out          # odd step keeps both E1 fill rules


def _run(policy, freq="Q"):
    res = run(_D["px"], _D["dist"], _D["start"], _D["end"], policy, cost=COST, freq=freq)
    t = res.totals
    left = t["contributed"] + t["div_gross"] - t["div_tax"]
    right = t["cash_end"] + t["spent_on_shares"] + t["fees"]
    assert abs(left - right) < 1e-6, f"accounting broken in {policy.name}"
    return res


def _job_grid(job):
    style, rule, pieces, deadline, tt = job
    m = summarize(_run(DipStrategy(rule, style, pieces, deadline, tt)), COST)
    m.update({"style": style, "rule": rule.name, "ref": rule.ref, "kind": rule.thr_kind, "thr": rule.thr,
              "reentry": rule.reentry if rule.reentry != "C" else f"C{rule.cooldown}",
              "pieces": pieces, "deadline": str(deadline), "trade_through_bp": tt * 1e4})
    return m


def pmap(fn, jobs, data, label, chunksize=16):
    t0, out = time.time(), []
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init, initargs=(data,)) as ex:
        for i, r in enumerate(ex.map(fn, jobs, chunksize=chunksize), 1):
            out.append(r)
            if i % 1000 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {label}: {i:,}/{len(jobs):,}  ({el:.0f}s, ~{el / i * (len(jobs) - i):.0f}s left)")
    return out


# ---------------------------------------------------------------- pre-flight
def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def preflight(args, dry_dir):
    problems = []
    for f in ["results/01_core_grid/runs.parquet", "results/07_plateau/finalists.csv", "notes/finalists.md"]:
        if not (ROOT / f).exists():
            problems.append(f"missing {f}")
    if args.holdout:
        if not (dry_dir / "ok.txt").exists():
            problems.append("run the rehearsal first:  python scripts/9_holdout.py --dry-run")
        if git("rev-parse", "--git-dir").returncode != 0:
            problems.append("not a git repository")
        else:
            for f in ["notes/finalists.md", "notes/decisions.md", "notes/predictions.md"]:
                if not git("ls-files", "--error-unmatch", f).returncode == 0:
                    problems.append(f"{f} is not committed")
                elif git("status", "--porcelain", "--", f).stdout.strip():
                    problems.append(f"{f} has uncommitted changes: commit it first")
        dec = ROOT / "notes" / "decisions.md"
        if not dec.exists() or "Holdout scope" not in dec.read_text():
            problems.append("notes/decisions.md has no 'Holdout scope' entry")
        log = ROOT / "notes" / "holdout_log.md"
        opened = [l for l in log.read_text().splitlines() if l.startswith("- ")] if log.exists() else []
        if opened and not args.rerun:
            problems.append(f"holdout already opened {len(opened)}x (see notes/holdout_log.md). "
                            "Only re-run after a crash, with --rerun; it gets logged too.")
    if problems:
        print("PRE-FLIGHT FAILED, holdout NOT opened:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("pre-flight OK")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--holdout", action="store_true")
    ap.add_argument("--rerun", action="store_true", help="only after a crash during the real run")
    ap.add_argument("--quick", action="store_true", help="dry-run only: 1 in 20 grid configs")
    args = ap.parse_args()
    if args.quick and args.holdout:
        sys.exit("--quick is for the rehearsal only")

    dry_dir = ROOT / "results" / "09_holdout_DRYRUN"
    preflight(args, dry_dir)
    if args.holdout:
        out, fig_dir, tag = ROOT / "results" / "09_holdout", ROOT / "figures", "HOLDOUT 2019 -> last day"
        px, dist, start, end = load("holdout", holdout=True)          # <- the one unlock, logged
    else:
        out, fig_dir, tag = dry_dir, dry_dir / "figures", "DRY RUN on 2014-2018 (build data, nothing unlocked)"
        px, dist, _, _ = load("E2_build")
        start, end = pd.Timestamp(DRY_START), pd.Timestamp(DRY_END)
    out.mkdir(parents=True, exist_ok=True)
    data = (px, dist, start, end)
    _init(data)
    dates = px.index[(px.index >= start) & (px.index <= end)]

    # ---------------- benchmarks + references
    rows, daily = [], {}
    for key, label, make, freq in BENCH:
        res = _run(make(), freq)
        m = summarize(res, COST)
        m.update({"key": key, "label": label, "arrival": freq})
        rows.append(m)
        daily[key] = res.daily["value"]
    fin = pd.read_csv(ROOT / "results" / "07_plateau" / "finalists.csv")
    for _, f in fin.iterrows():
        key = f"ref_{f['style']}"
        res = _run(ref_policy(f))
        m = summarize(res, COST)
        m.update({"key": key, "label": f"{f['style']} {f['label'].split(' (')[0].lower()}: {f['rule']} P{int(f['pieces'])} D{f['deadline']}",
                  "arrival": "Q", "build_d_xirr_bh": f["d_xirr_bh"], "style": f["style"], "fin_label": f["label"]})
        rows.append(m)
        daily[key] = res.daily["value"]
    bench = pd.DataFrame(rows).set_index("key")
    bh_q, bh_y = bench.loc["bh_q", "xirr_before_sell"], bench.loc["bh_y", "xirr_before_sell"]
    bench["d_vs_quarterly_bh"] = bench["xirr_before_sell"] - bh_q
    bench["d_vs_own_arrival"] = bench["xirr_before_sell"] - np.where(bench["arrival"] == "A", bh_y, bh_q)

    # ---------------- the whole grid (distribution only)
    jobs = grid_jobs(args.quick)
    print(f"running the grid: {len(jobs):,} configurations on {WORKERS} cores")
    grid = pd.DataFrame(pmap(_job_grid, jobs, data, "grid"))
    grid["d_xirr_bh"] = grid["xirr_before_sell"] - bh_q
    grid["d_xirr_dca"] = grid["xirr_before_sell"] - bench.loc["dca", "xirr_before_sell"]
    build = pd.read_parquet(ROOT / "results" / "01_core_grid" / "runs.parquet")
    k = ["style", "rule", "pieces", "deadline", "trade_through_bp"]
    for df in (grid, build):
        df["deadline"] = df["deadline"].astype(str)
        df["trade_through_bp"] = df["trade_through_bp"].astype(float).round(1)
        df["pieces"] = df["pieces"].astype(int)
    grid = grid.merge(build[k + ["d_xirr_bh"]].rename(columns={"d_xirr_bh": "build_d_xirr_bh"}), on=k, how="left")
    grid["primary"] = grid["trade_through_bp"] == grid["style"].map(PRIMARY_TT)
    # break-even formula on this period
    contrib = contribution_schedule(dates, start, end)
    ybar = np.mean([(end - d).days / 365.25 for d in contrib])
    grid["pred_d_bh"] = -(bh_q * grid["avg_wait_days"] / 365
                          + (grid["fees_pct_of_contrib"] - bench.loc["bh_q", "fees_pct_of_contrib"])) / ybar
    grid["timing_skill"] = grid["d_xirr_bh"] - grid["pred_d_bh"]

    # ---------------- save
    bench.reset_index().to_csv(out / "benchmarks.csv", index=False)
    grid.to_csv(out / "grid.csv", index=False)
    if args.holdout:
        master = ROOT / "results" / "master.csv"
        old = pd.read_csv(master, low_memory=False) if master.exists() else pd.DataFrame()
        if len(old):
            old = old[old["study"] != "09_holdout"]
        new = pd.concat([bench.reset_index().assign(study="09_holdout"), grid.assign(study="09_holdout")], ignore_index=True)
        pd.concat([old, new], ignore_index=True).to_csv(master, index=False)

    # ---------------- print
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    section(tag)
    say(f"period {dates[0].date()} -> {dates[-1].date()}  ({(dates[-1] - dates[0]).days / 365.25:.1f} years), "
        f"EUR 2,000/yr, IBKR Tiered, whole shares")
    say("\nBENCHMARKS (vs quarterly B&H; yearly lines also vs yearly B&H)")
    for key, r in bench.iterrows():
        if key.startswith("ref_"):
            continue
        extra = f" | vs yearly B&H {100 * r['d_vs_own_arrival']:+.2f}" if r["arrival"] == "A" else ""
        say(f"  {r['label']:46s} XIRR {r['xirr_before_sell']:6.2%} (after sell {r['xirr_after_sell']:6.2%}) | "
            f"vs quarterly B&H {100 * r['d_vs_quarterly_bh']:+.2f}{extra} | contributed {r['contributed']:,.0f} | "
            f"final {r['final_value']:,.0f} | fees {r['fees_pct_of_contrib']:.2%} | max DD {r['max_drawdown']:.1%}")
    say("\nREFERENCE DIP CONFIGS (NOT QUALIFIED on the build period: shown for transparency, can't change the verdict)")
    for key, r in bench[bench.index.str.startswith("ref_")].iterrows():
        verdict = "beats" if r["d_vs_quarterly_bh"] > 0 else "loses to"
        say(f"  {r['label']}")
        say(f"     build {100 * r['build_d_xirr_bh']:+.2f} -> holdout {100 * r['d_vs_quarterly_bh']:+.2f} pts/yr "
            f"({verdict} B&H) | XIRR {r['xirr_before_sell']:.2%} | fees {r['fees_pct_of_contrib']:.2%} | "
            f"wait {r['avg_wait_days']:.0f}d | buys {r['n_buys']}")
    say("\nWHOLE GRID, DISTRIBUTION ONLY (primary fill; nothing is selected from this)")
    for s in ("E1", "E2"):
        d = grid[(grid["style"] == s) & grid["primary"]]
        b = d["build_d_xirr_bh"].dropna()
        h = d["d_xirr_bh"]
        both = d.dropna(subset=["build_d_xirr_bh"])
        say(f"  {s}: {len(d):,} configs")
        say(f"     beat B&H     build {(b > 0).mean():6.1%}  ->  holdout {(h > 0).mean():6.1%}  ({int((h > 0).sum())} configs)")
        say(f"     median       build {100 * b.median():+.2f}  ->  holdout {100 * h.median():+.2f} pts/yr "
            f"(holdout 10% {100 * h.quantile(.1):+.2f} | 90% {100 * h.quantile(.9):+.2f})")
        say(f"     beat DCA     holdout {(d['d_xirr_dca'] > 0).mean():.1%}")
        say(f"     build vs holdout, same config: corr {both['build_d_xirr_bh'].corr(both['d_xirr_bh']):.2f} | "
            f"rank corr {both['build_d_xirr_bh'].rank().corr(both['d_xirr_bh'].rank()):.2f}")
        say(f"     break-even formula: corr(actual, predicted) {h.corr(d['pred_d_bh']):.2f} | "
            f"timing skill median {100 * d['timing_skill'].median():+.2f} pts/yr")
    if not args.quick:
        t = grid[grid["style"] == "E1"].groupby("trade_through_bp")["d_xirr_bh"].median() * 100
        say(f"  E1 fill rule, median: touch {t.get(0.0, np.nan):+.2f} | 0.05% through {t.get(5.0, np.nan):+.2f}")
    say("\nVERDICT (pre-registered, §9-10): no rule qualified on the build period, so there was no finalist to test.")
    say("The conclusion 'no evidence these dip rules beat investing on arrival' stands; the numbers above say")
    say("whether the fresh data agrees.")

    charts(bench, grid, daily, fig_dir, dates, args.holdout)
    if args.holdout:
        md = ROOT / "notes" / "holdout_results.md"
        md.write_text("# Holdout results (generated by scripts/9_holdout.py)\n\n"
                      f"Run on {pd.Timestamp.now():%Y-%m-%d %H:%M}. Unlock logged in notes/holdout_log.md.\n\n"
                      "```\n" + "\n".join(lines) + "\n```\n")
    elif not args.quick:
        (out / "ok.txt").write_text(f"dry run passed {pd.Timestamp.now():%Y-%m-%d %H:%M}\n")
    section("WROTE")
    print(f"  {out.relative_to(ROOT)}/benchmarks.csv, grid.csv")
    print(f"  {fig_dir.relative_to(ROOT)}/fig11..fig14")
    if args.holdout:
        print("  notes/holdout_results.md  <- commit it with the results")
    else:
        print("  rehearsal passed." + ("" if args.quick else " When ready:  python scripts/9_holdout.py --holdout"))
    print("DONE. Paste everything back.")


# ---------------------------------------------------------------- charts
def _spread_labels(ys, gap):
    """Push right-edge labels apart so they never overlap (keeps their order)."""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def charts(bench, grid, daily, fig_dir, dates, holdout):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    plots.style()
    C = plots.SERIES
    better, worse = "#2a78d6", "#e34948"
    span = f"{dates[0].year}-{dates[-1].year}"
    pre = "" if holdout else "DRY RUN  "

    # fig11 everything vs quarterly B&H
    items = [(r["label"], r["d_vs_quarterly_bh"] * 100, "hind" if k.startswith("ceil") else "x")
             for k, r in bench.iterrows() if k != "bh_q"]
    for s in ("E1", "E2"):
        d = grid[(grid["style"] == s) & grid["primary"]]["d_xirr_bh"] * 100
        items.append((f"{s} dip grid: median of all {len(d):,} configs", d.median(), "x"))
    items.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(items) + 1.6))
    y = np.arange(len(items))
    v = np.array([t[1] for t in items])
    cols = [C[2] if t[2] == "hind" else (better if t[1] > 0 else worse) for t in items]
    ax.barh(y, v, color=cols, height=0.6)
    ax.axvline(0, color=plots.INK, linewidth=1)
    lo_, hi_ = min(-0.3, v.min()), max(0.3, v.max())
    pad = 0.14 * (hi_ - lo_)
    lim = hi_ - lo_
    ax.set_xlim(lo_ - pad, hi_ + pad)
    for yi, x in zip(y, v):
        ax.text(x + (0.015 * lim if x >= 0 else -0.015 * lim), yi, f"{x:+.2f}", va="center",
                ha="left" if x >= 0 else "right", fontsize=8, color=plots.INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([t[0] for t in items], fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("XIRR minus quarterly Buy & Hold (pts/yr)")
    ax.set_title(f"{pre}{span}: everything vs quarterly Buy & Hold ({bench.loc['bh_q', 'xirr_before_sell']:.2%}/yr)")
    plots.note(fig, "Green = hindsight (not a strategy). Yearly lines receive EUR 2,000 every January instead of EUR 500 a quarter. "
                    "Reference configs did NOT qualify on the build period.")
    plots.save(fig, fig_dir / "fig11_holdout_summary.png")

    # fig12 grid distribution, holdout vs build
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    lo, hi = -1.5, 1.0
    bins = np.linspace(lo, hi, 61)
    for ax, s in zip(axes, ("E1", "E2")):
        d = grid[(grid["style"] == s) & grid["primary"]]
        h = (d["d_xirr_bh"] * 100).clip(lo, hi)
        b = (d["build_d_xirr_bh"].dropna() * 100).clip(lo, hi)
        ax.hist(h, bins=bins, color=C[0], edgecolor=plots.SURFACE, linewidth=0.5)
        ax.hist(b, bins=bins, histtype="step", color=plots.INK2, linewidth=1.2)
        for x, col in [(0, plots.INK), (bench.loc["dca", "d_vs_quarterly_bh"] * 100, C[1]),
                       (bench.loc["ceil_q", "d_vs_quarterly_bh"] * 100, C[2])]:
            ax.axvline(min(x, hi), color=col, linewidth=1.4)
        top = ax.get_ylim()[1]
        ax.text(0, top * 0.97, " Buy & Hold", color=plots.INK, fontsize=8, va="top")
        ax.text(bench.loc["dca", "d_vs_quarterly_bh"] * 100, top * 0.87, "DCA ", color=C[1], fontsize=8, va="top", ha="right")
        ax.text(min(bench.loc["ceil_q", "d_vs_quarterly_bh"] * 100, hi), top * 0.97, "ceiling ", color=C[2], fontsize=8,
                va="top", ha="right")
        ax.set_title(f"{s}: {(d['d_xirr_bh'] > 0).mean():.0%} beat B&H on {span} vs "
                     f"{(d['build_d_xirr_bh'] > 0).mean():.0%} on the build period", fontsize=10)
        ax.set_xlabel("strategy XIRR minus Buy & Hold (pts/yr)")
        ax.set_ylabel("configurations")
    plots.note(fig, f"{pre}Blue bars = {span}. Grey outline = the same configs on the build period. Values beyond the "
                    f"axis are piled at the edges. Primary fill rule.")
    plots.save(fig, fig_dir / "fig12_holdout_grid.png")

    # fig13 build vs holdout, same config
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, s in zip(axes, ("E1", "E2")):
        d = grid[(grid["style"] == s) & grid["primary"]].dropna(subset=["build_d_xirr_bh"])
        x = (d["build_d_xirr_bh"] * 100).clip(-2, 0.5)
        yv = (d["d_xirr_bh"] * 100).clip(-2, 0.5)
        ax.scatter(x, yv, s=6, color=C[0], alpha=0.35, linewidth=0)
        ax.axhline(0, color=plots.INK, linewidth=0.8)
        ax.axvline(0, color=plots.INK, linewidth=0.8)
        ax.plot([-2, 0.5], [-2, 0.5], color=plots.INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xlim(-2, 0.5)
        ax.set_ylim(-2, 0.5)
        n_tr = int(((d["build_d_xirr_bh"] > 0) & (d["d_xirr_bh"] > 0)).sum())
        ax.text(0.04, 0.3, f"won both:\n{n_tr}", fontsize=8, color=plots.INK2)
        ax.set_xlabel("build period (pts/yr vs B&H)")
        ax.set_ylabel(f"{span} (pts/yr vs B&H)")
        ax.set_title(f"{s}: same config, past vs fresh data (corr {d['build_d_xirr_bh'].corr(d['d_xirr_bh']):.2f})")
    plots.note(fig, "Each dot is one configuration. Top-right box = beat B&H in both periods. Dashed = same result in both. "
                    "Clipped to [-2, +0.5].")
    plots.save(fig, fig_dir / "fig13_holdout_build_vs_holdout.png")

    # fig14 account value vs quarterly B&H over time (quarterly money only)
    base = daily["bh_q"]
    keys = ["dca", "save_up", "ceil_q"] + [k for k in daily if k.startswith("ref_")]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axhline(0, color=plots.INK, linewidth=1)
    for a, b, lab in EVENTS:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if a >= base.index[0] and b <= base.index[-1]:
            ax.axvspan(a, b, color=plots.GRID, alpha=0.6, linewidth=0)
            ax.text(a, 1.0, f" {lab}", transform=ax.get_xaxis_transform(), fontsize=7.5, color=plots.MUTED, va="top")
    ends, labs, colrs = [], [], []
    colour = {"dca": C[1], "ceil_q": C[2], "save_up": C[3], "ref_E1": C[4], "ref_E2": C[6]}
    for key in keys:
        rel = (daily[key] / base - 1) * 100
        col = colour.get(key, C[7])
        ax.plot(rel.index, rel.values, color=col, linewidth=1.6)
        ends.append(rel.values[-1])
        labs.append(bench.loc[key, "label"])
        colrs.append(col)
    yl = ax.get_ylim()
    for yv, lab, col in zip(_spread_labels(ends, (yl[1] - yl[0]) * 0.06), labs, colrs):
        ax.text(base.index[-1], yv, f"  {lab}", color=col, fontsize=8, va="center")
    ax.set_xlim(base.index[0], base.index[-1] + (base.index[-1] - base.index[0]) * 0.55)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xticks([t for t in ax.get_xticks() if mdates.date2num(base.index[0]) <= t <= mdates.date2num(base.index[-1])])
    ax.spines["bottom"].set_bounds(mdates.date2num(base.index[0]), mdates.date2num(base.index[-1]))
    ax.set_ylabel("account value vs quarterly B&H (%)")
    ax.set_title(f"{pre}{span}: account value relative to quarterly Buy & Hold (0 = same)")
    plots.note(fig, "Same EUR 500 a quarter for everyone. The first year is noisy because the account is still tiny. "
                    "Yearly-money lines are left out (different contributions).")
    plots.save(fig, fig_dir / "fig14_holdout_over_time.png")


if __name__ == "__main__":
    main()
