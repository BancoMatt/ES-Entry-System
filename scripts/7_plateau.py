"""
7_plateau.py  -  pre-registration §9: does any dip rule REALLY beat Buy & Hold?

Uses the 12,720 build-period results from 6_core_grid.py, then applies the frozen rules
in order (E1 uses the primary fill = 0.05% trade-through; touch is shown in the grid only):

  1. candidate  the rule is in notes/candidates_<style>.csv (fires 8-12x/yr, >= 2 in worst year)
  2. beats      d_xirr_bh > 0
  3. plateau    >= 75% of its NEIGHBOURS also beat B&H
                neighbours = drop size one step up/down, pieces +-1, deadline one step
                (deadline order by longest possible wait: 30 < 60 < next < none)
                60-75% = "near-plateau": shown, never selected
  4. halves     beats B&H in BOTH halves of the build period, run separately
                E1: 2009-13 | 2014-18     E2: 1999-2008 | 2009-18
  5. bootstrap  beats B&H in >= 70% of 500 alternative histories
                (63-day blocks of real daily bars reshuffled; no dividends for anyone)

  Finalists: at most one per re-entry type (A, B, C) per style; family = reference x re-entry
  x fixed/vol; ranked by SHARE of the family's configs that qualify; finalist = the qualifying
  config closest to the family's MEDIAN result. Nothing qualifies -> the family with the most
  configs beating B&H sends one LABELLED, NOT-QUALIFIED config to the holdout.

Also: overfitting check (PBO via CSCV, 8 blocks) on the candidate set, the same plateau
screen vs monthly DCA (secondary, steps 1-4), and all charts in figures/.

Outputs: results/07_plateau/*.csv, notes/finalists.md, figures/fig01..fig07*.png
Run from project root:  python scripts/7_plateau.py
"""
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import plots                                                 # noqa: E402
from src.benchmarks import BuyHold, MonthlyDCA, PerfectTimingCeiling  # noqa: E402
from src.costs import IBKR_TIERED                                     # noqa: E402
from src.data import load                                             # noqa: E402
from src.engine import contribution_schedule, run                    # noqa: E402
from src.metrics import summarize, xirr                              # noqa: E402
from src.signals import Rule                                          # noqa: E402
from src.strategy import DipStrategy                                  # noqa: E402

PERIOD = {"E1": "E1_build", "E2": "E2_build"}
HALVES = {"E1": [("2009-01-02", "2013-12-31"), ("2014-01-01", "2018-12-31")],
          "E2": [("1999-01-04", "2008-12-31"), ("2009-01-01", "2018-12-31")]}
PRIMARY_TT = {"E1": 5.0, "E2": 0.0}                  # basis points
THR = {"fixed": [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05],
       "vol": [1.0, 1.5, 2.0, 2.5, 3.0]}
DL_ORDER = ["30", "60", "next", "none"]
PIECES = [1, 2, 3, 4]
SHARE_Q, SHARE_NEAR, BOOT_MIN = 0.75, 0.60, 0.70
BOOT_PATHS = int(os.environ.get("BOOT_PATHS", 500))     # env override only for quick tests
BLOCK = 63
TYPE_OF = {"A": "A", "B": "B", "C5": "C", "C20": "C"}
OUT, FIG, NOTES = ROOT / "results" / "07_plateau", ROOT / "figures", ROOT / "notes"
WORKERS = max(1, (os.cpu_count() or 2) - 1)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 100)


def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# ======================================================================= helpers
def rule_of(r):
    re_ = r["reentry"]
    if isinstance(re_, str) and re_.startswith("C"):
        return Rule(r["ref"], r["kind"], float(r["thr"]), "C", cooldown=int(re_[1:]))
    return Rule(r["ref"], r["kind"], float(r["thr"]), re_)


def deadline_of(d):
    return int(d) if str(d).isdigit() else str(d)


def policy_of(style, r):
    return DipStrategy(rule_of(r), style, int(r["pieces"]), deadline_of(r["deadline"]),
                       float(r["trade_through_bp"]) / 1e4)


def cfg_key(r):
    return (r["style"], r["ref"], r["reentry"], r["kind"], round(float(r["thr"]), 6),
            int(r["pieces"]), str(r["deadline"]))


def neighbours(k):
    style, ref, re_, kind, thr, p, d = k
    out = []
    ts = THR[kind]
    i = int(np.argmin([abs(t - thr) for t in ts]))
    for j in (i - 1, i + 1):
        if 0 <= j < len(ts):
            out.append((style, ref, re_, kind, round(ts[j], 6), p, d))
    for q in (p - 1, p + 1):
        if q in PIECES:
            out.append((style, ref, re_, kind, thr, q, d))
    di = DL_ORDER.index(d)
    for j in (di - 1, di + 1):
        if 0 <= j < len(DL_ORDER):
            out.append((style, ref, re_, kind, thr, p, DL_ORDER[j]))
    return out


def twr_returns(daily):
    ext = daily["contributed"].diff().fillna(daily["contributed"])
    prev = daily["value"].shift(1)
    return ((daily["value"] - ext) / prev - 1).where(prev > 0).fillna(0.0).to_numpy(np.float64)


def boot_prices(px, start, end, seed):
    """Alternative history: real data before `start`, then 63-day blocks of real daily bars
    (open/high/low/close relative to the previous close) drawn at random and chained."""
    rng = np.random.default_rng(seed)
    hist = px[px.index < start]
    per = px[(px.index >= start) & (px.index <= end)]
    prevc = px["close"].shift(1).loc[per.index].fillna(per["close"].iloc[0])
    rel = per[["open", "high", "low", "close"]].div(prevc, axis=0).to_numpy()
    n = len(rel)
    out = np.empty_like(rel)
    pos = 0
    while pos < n:
        s = rng.integers(0, n - BLOCK + 1)
        take = min(BLOCK, n - pos)
        out[pos:pos + take] = rel[s:s + take]
        pos += take
    c0 = hist["close"].iloc[-1] if len(hist) else per["close"].iloc[0]
    closes = c0 * np.cumprod(out[:, 3])
    pc = np.r_[c0, closes[:-1]]
    new = per.copy()
    new["open"], new["high"], new["low"], new["close"] = out[:, 0] * pc, out[:, 1] * pc, out[:, 2] * pc, closes
    return pd.concat([hist, new])


# ======================================================================= workers
_DATA = {}
NO_DIST = pd.DataFrame(columns=["ex_date", "pay_date", "amount_eur"])


def _init():
    sys.path.insert(0, str(ROOT))
    for s, p in PERIOD.items():
        _DATA[s] = load(p)


def _job_halves(job):
    style, row = job
    px, dist, _, _ = _DATA[style]
    out = []
    for a, b in HALVES[style]:
        res = run(px, dist, pd.Timestamp(a), pd.Timestamp(b), policy_of(style, row))
        out.append(summarize(res, IBKR_TIERED)["xirr_before_sell"])
    return out


def _job_boot(job):
    style, seed, rows = job
    px, _, start, end = _DATA[style]
    bp = boot_prices(px, start, end, seed)
    bh = xirr_of(run(bp, NO_DIST, start, end, BuyHold()))
    return [xirr_of(run(bp, NO_DIST, start, end, policy_of(style, r))) - bh for r in rows]


def _job_daily(job):
    style, row = job
    px, dist, start, end = _DATA[style]
    return twr_returns(run(px, dist, start, end, policy_of(style, row)).daily).astype(np.float32)


def xirr_of(res):
    t = res.totals
    return xirr(res.flows + [(t["end_date"], t["value_end"])])


def pmap(fn, jobs, label):
    t0 = time.time()
    out = []
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init) as ex:
        for k, r in enumerate(ex.map(fn, jobs, chunksize=max(1, len(jobs) // (WORKERS * 8) or 1)), 1):
            out.append(r)
            if k % max(1, len(jobs) // 5) == 0 or k == len(jobs):
                print(f"    {label}: {k:,}/{len(jobs):,}  ({time.time()-t0:.0f}s)")
    return out


# ======================================================================= PBO (CSCV)
def pbo_cscv(R, n_blocks=8):
    """R: days x configs matrix of daily excess returns vs B&H. Returns (pbo, logits)."""
    T, N = R.shape
    blocks = np.array_split(np.arange(T), n_blocks)
    perf = np.stack([R[b].mean(axis=0) for b in blocks])           # n_blocks x N
    logits = []
    for ins in itertools.combinations(range(n_blocks), n_blocks // 2):
        oos = [b for b in range(n_blocks) if b not in ins]
        is_perf, oos_perf = perf[list(ins)].mean(0), perf[oos].mean(0)
        best = int(np.argmax(is_perf))
        rank = (oos_perf < oos_perf[best]).sum() + 0.5 * ((oos_perf == oos_perf[best]).sum() - 1) + 1
        w = rank / (N + 1)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return float((logits <= 0).mean()), logits


# ======================================================================= main
def main():
    for d in (OUT, FIG):
        d.mkdir(parents=True, exist_ok=True)
    plots.style()
    df = pd.read_parquet(ROOT / "results" / "01_core_grid" / "runs.parquet")
    refs = pd.read_csv(ROOT / "results" / "01_core_grid" / "reference_lines.csv")
    cands = {s: set(pd.read_csv(NOTES / f"candidates_{s}.csv")["rule"]) for s in PERIOD}
    df["deadline"] = df["deadline"].astype(str)
    df["candidate"] = [r in cands[s] for s, r in zip(df["style"], df["rule"])]
    prim = df[df["trade_through_bp"] == df["style"].map(PRIMARY_TT)].copy()
    prim["key"] = [cfg_key(r) for _, r in prim.iterrows()]
    _init()

    # ---------------- reference numbers per style
    ref = {}
    for s in PERIOD:
        rr = refs[refs["style"] == s].set_index("policy")
        px, dist, start, end = _DATA[s]
        contrib = contribution_schedule(px.index[(px.index >= start) & (px.index <= end)], start, end)
        ybar = np.mean([(end - d).days / 365.25 for d in contrib])
        ref[s] = {"bh": rr.loc["buy_hold", "xirr_before_sell"],
                  "dca": rr.loc["dca_monthly_d01", "xirr_before_sell"],
                  "ceil": rr.loc["ceiling_perfect_timing", "xirr_before_sell"],
                  "bh_fee": rr.loc["buy_hold", "fees_pct_of_contrib"], "ybar": ybar}

    # ---------------- steps 1-3: neighbour shares
    for col, tag in (("d_xirr_bh", "bh"), ("d_xirr_dca", "dca")):
        lookup = dict(zip(prim["key"], prim[col]))
        shares = []
        for k in prim["key"]:
            nb = [lookup[n] for n in neighbours(k) if n in lookup]
            shares.append(np.mean([v > 0 for v in nb]) if nb else np.nan)
        prim[f"share_{tag}"] = shares
        prim[f"screen_{tag}"] = prim["candidate"] & (prim[col] > 0) & (prim[f"share_{tag}"] >= SHARE_Q)
        prim[f"near_{tag}"] = prim["candidate"] & (prim[col] > 0) & prim[f"share_{tag}"].between(SHARE_NEAR, SHARE_Q, inclusive="left")

    # ---------------- step 4: halves (for anything that passed 1-3 vs B&H or vs DCA)
    need = prim[prim["screen_bh"] | prim["screen_dca"]]
    section(f"STEPS 1-3 passed: vs B&H {int(prim['screen_bh'].sum())} | vs DCA {int(prim['screen_dca'].sum())} "
            f"-> running both halves for {len(need)} configs")
    halves_ref = {}
    for s in PERIOD:
        px, dist, _, _ = _DATA[s]
        for a, b in HALVES[s]:
            for pol in (BuyHold(), MonthlyDCA(1)):
                r_ = run(px, dist, pd.Timestamp(a), pd.Timestamp(b), pol)
                halves_ref[(s, a, pol.name)] = summarize(r_, IBKR_TIERED)["xirr_before_sell"]
    if len(need):
        hv = pmap(_job_halves, [(r["style"], r) for _, r in need.iterrows()], "halves")
        for (idx, r), (x1, x2) in zip(need.iterrows(), hv):
            (a1, _), (a2, _) = HALVES[r["style"]]
            prim.loc[idx, "h1_d_bh"] = x1 - halves_ref[(r["style"], a1, "buy_hold")]
            prim.loc[idx, "h2_d_bh"] = x2 - halves_ref[(r["style"], a2, "buy_hold")]
            prim.loc[idx, "h1_d_dca"] = x1 - halves_ref[(r["style"], a1, "dca_monthly_d01")]
            prim.loc[idx, "h2_d_dca"] = x2 - halves_ref[(r["style"], a2, "dca_monthly_d01")]
    for tag in ("bh", "dca"):
        for c in (f"h1_d_{tag}", f"h2_d_{tag}"):
            if c not in prim:
                prim[c] = np.nan
        prim[f"halves_{tag}"] = prim[f"screen_{tag}"] & (prim[f"h1_d_{tag}"] > 0) & (prim[f"h2_d_{tag}"] > 0)

    # ---------------- step 5: bootstrap (vs B&H only, as pre-registered)
    prim["boot_p"] = np.nan
    boot_store = {}
    for s in PERIOD:
        rows = prim[(prim["style"] == s) & prim["halves_bh"]]
        if not len(rows):
            continue
        section(f"STEP 5 bootstrap {s}: {len(rows)} configs x {BOOT_PATHS} alternative histories")
        res = pmap(_job_boot, [(s, seed, [r for _, r in rows.iterrows()]) for seed in range(BOOT_PATHS)], "bootstrap")
        mat = np.array(res)                                           # paths x configs
        for j, idx in enumerate(rows.index):
            prim.loc[idx, "boot_p"] = float((mat[:, j] > 0).mean())
            boot_store[idx] = mat[:, j]
    prim["qualifies"] = prim["halves_bh"] & (prim["boot_p"] >= BOOT_MIN)

    # ---------------- finalists
    finalists = []
    for s in PERIOD:
        ps = prim[prim["style"] == s]
        fam_cols = ["ref", "reentry", "kind"]
        fam = ps.groupby(fam_cols).agg(n=("key", "size"), q=("qualifies", "sum"),
                                       beat=("d_xirr_bh", lambda x: int((x > 0).sum()))).reset_index()
        fam["share"] = fam["q"] / fam["n"]
        fam["type"] = fam["reentry"].map(TYPE_OF)
        picked = False
        for t in ("A", "B", "C"):
            ft = fam[(fam["type"] == t) & (fam["q"] > 0)].sort_values(["share", "q", "kind"], ascending=[False, False, True])
            if not len(ft):
                continue
            f = ft.iloc[0]
            qs = ps[(ps["ref"] == f["ref"]) & (ps["reentry"] == f["reentry"]) & (ps["kind"] == f["kind"]) & ps["qualifies"]]
            med = qs["d_xirr_bh"].median()
            pick = qs.assign(dist=(qs["d_xirr_bh"] - med).abs()).sort_values(["dist", "fees", "pieces"]).iloc[0]
            finalists.append({**pick.to_dict(), "label": "QUALIFIED", "family_share": f["share"], "type": t})
            picked = True
        typed = fam[fam["type"].notna() & (fam["beat"] > 0)]
        if not picked and len(typed):
            f = typed.sort_values(["beat", "share"], ascending=False).iloc[0]
            beat = ps[(ps["ref"] == f["ref"]) & (ps["reentry"] == f["reentry"]) & (ps["kind"] == f["kind"]) & (ps["d_xirr_bh"] > 0)]
            if len(beat):
                med = beat["d_xirr_bh"].median()
                pick = beat.assign(dist=(beat["d_xirr_bh"] - med).abs()).sort_values(["dist", "fees"]).iloc[0]
                finalists.append({**pick.to_dict(), "label": "NOT QUALIFIED (reference only)",
                                  "family_share": 0.0, "type": f["type"]})
    fin = pd.DataFrame(finalists)

    # ---------------- PBO on the candidate set
    pbo = {}
    for s in PERIOD:
        cs = prim[(prim["style"] == s) & prim["candidate"]]
        section(f"OVERFITTING CHECK (PBO, CSCV 8 blocks) {s}: {len(cs)} candidate configs")
        if len(cs) < 2:
            pbo[s] = (np.nan, np.array([]))
            continue
        px, dist, start, end = _DATA[s]
        bh_r = twr_returns(run(px, dist, start, end, BuyHold()).daily)
        R = np.column_stack(pmap(_job_daily, [(s, r) for _, r in cs.iterrows()], "daily runs")) - bh_r[:, None]
        pbo[s] = pbo_cscv(R)

    # ---------------- break-even (timing skill = actual - predicted)
    for s in PERIOD:
        m = prim["style"] == s
        r_ = ref[s]
        pred = -(r_["bh"] * prim.loc[m, "avg_wait_days"] / 365 + (prim.loc[m, "fees_pct_of_contrib"] - r_["bh_fee"])) / r_["ybar"]
        prim.loc[m, "pred_d_bh"] = pred
        prim.loc[m, "timing_skill"] = prim.loc[m, "d_xirr_bh"] - pred

    # ---------------- save tables
    keep = ["style", "rule", "ref", "reentry", "kind", "thr", "pieces", "deadline", "trade_through_bp",
            "candidate", "xirr_before_sell", "d_xirr_bh", "d_xirr_dca", "ceiling_share", "share_bh",
            "screen_bh", "near_bh", "h1_d_bh", "h2_d_bh", "halves_bh", "boot_p", "qualifies",
            "share_dca", "screen_dca", "halves_dca", "avg_wait_days", "fees_pct_of_contrib",
            "pred_d_bh", "timing_skill", "n_buys", "max_drawdown"]
    prim[keep].to_csv(OUT / "plateau.csv", index=False)
    if len(fin):
        fin[[c for c in keep if c in fin] + ["label", "family_share", "type"]].to_csv(OUT / "finalists.csv", index=False)

    # ======================================================================= PRINT
    for s in PERIOD:
        ps = prim[prim["style"] == s]
        r_ = ref[s]
        section(f"{s} ({PERIOD[s]})  B&H {r_['bh']:.2%} | DCA {r_['dca']:.2%} | ceiling {r_['ceil']:.2%}")
        print(f"configs (primary fill)           {len(ps):,}")
        print(f"1 candidate rule                 {int(ps['candidate'].sum()):,}")
        print(f"2 ... and beats B&H              {int((ps['candidate'] & (ps['d_xirr_bh'] > 0)).sum()):,}")
        print(f"3 ... and >= 75% neighbours win  {int(ps['screen_bh'].sum()):,}   "
              f"(near-plateau 60-75%: {int(ps['near_bh'].sum()):,})")
        print(f"4 ... and wins BOTH halves       {int(ps['halves_bh'].sum()):,}")
        print(f"5 ... and >= 70% of bootstraps   {int(ps['qualifies'].sum()):,}   <- QUALIFIED")
        p, lg = pbo[s]
        print(f"\nPBO (probability the best in-sample config is below median out-of-sample): {p:.0%}"
              f"   (0% = no overfitting risk, 50% = selection is a coin flip)")
        sk = ps["timing_skill"] * 100
        print(f"break-even check: actual minus formula prediction (= timing skill), pts/yr: "
              f"median {sk.median():+.2f} | 10% {sk.quantile(.1):+.2f} | 90% {sk.quantile(.9):+.2f} | "
              f"corr(actual, predicted) {ps['d_xirr_bh'].corr(ps['pred_d_bh']):.2f}")
        top = ps[ps["candidate"] & (ps["d_xirr_bh"] > 0)].sort_values("d_xirr_bh", ascending=False).head(10)
        if len(top):
            print("\ncandidate configs that beat B&H (top 10 by result, all shown in plateau.csv):")
            t = top[["rule", "pieces", "deadline", "d_xirr_bh", "share_bh", "h1_d_bh", "h2_d_bh", "boot_p", "qualifies"]].copy()
            for c in ["d_xirr_bh", "h1_d_bh", "h2_d_bh"]:
                t[c] = (t[c] * 100).round(3)
            t["share_bh"] = (t["share_bh"] * 100).round(0)
            print(t.to_string(index=False))
        print(f"\nsecondary, same screen vs monthly DCA (steps 1-4): passed {int(ps['halves_dca'].sum()):,}")

    section("FINALISTS FOR THE HOLDOUT")
    if len(fin):
        f = fin[["style", "type", "rule", "pieces", "deadline", "d_xirr_bh", "family_share", "boot_p", "label"]].copy()
        f["d_xirr_bh"] = (f["d_xirr_bh"] * 100).round(3)
        print(f.to_string(index=False))
    else:
        print("none")

    # ======================================================================= CHARTS
    charts(prim, ref, pbo, boot_store, fin)
    write_finalists_md(fin, prim, pbo)
    section("WROTE")
    print("  results/07_plateau/plateau.csv, finalists.csv | notes/finalists.md | figures/fig01..fig07*.png")
    print("DONE. Paste everything back.")


# ======================================================================= charts
def charts(prim, ref, pbo, boot_store, fin):
    import matplotlib.pyplot as plt
    C = plots.SERIES
    styles = list(PERIOD)

    # fig01 distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, s in zip(axes, styles):
        d = prim.loc[prim["style"] == s, "d_xirr_bh"] * 100
        ceil_x = (ref[s]["ceil"] - ref[s]["bh"]) * 100
        lo, hi = -1.5, max(0.3, ceil_x + 0.15)
        ax.hist(d.clip(lo, hi), bins=np.linspace(lo, hi, 73), color=C[0], edgecolor=plots.SURFACE, linewidth=0.5)
        for x, lab, col in [(0, "Buy & Hold", plots.INK), ((ref[s]["dca"] - ref[s]["bh"]) * 100, "monthly DCA", C[1]),
                            (ceil_x, "perfect-timing ceiling (hindsight)", C[2])]:
            ax.axvline(x, color=col, linewidth=1.5, label=f"{lab}  {x:+.2f}")
        ax.legend(loc="upper left")
        n_out = int((d < lo).sum())
        ax.set_title(f"{s}: {len(d):,} configurations vs Buy & Hold")
        ax.set_xlabel("strategy XIRR minus Buy & Hold (pts/yr)")
        ax.set_ylabel("number of configurations")
        ax.text(0.01, 0.55, f"{n_out:,} below {lo} piled at left edge\n{int((d > 0).sum())} beat Buy & Hold",
                transform=ax.transAxes, fontsize=8, color=plots.INK2)
    plots.note(fig, "Primary fill rule (E1: 0.05% trade-through). Build periods only. Source: results/01_core_grid/runs.csv")
    plots.save(fig, FIG / "fig01_distribution.png")

    # fig02 what matters
    dials = [("pieces", "pieces"), ("deadline", "deadline"), ("reentry", "re-entry"), ("ref", "reference"), ("kind", "drop style")]
    rows, labels, groups = [], [], []
    for col, nice in dials:
        vals = sorted(prim[col].astype(str).unique(), key=lambda v: (DL_ORDER.index(v) if v in DL_ORDER else 99, v))
        for v in vals:
            rows.append([prim.loc[(prim["style"] == s) & (prim[col].astype(str) == v), "d_xirr_bh"].median() * 100 for s in styles])
            labels.append(f"{nice}: {v}")
            groups.append(nice)
    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(labels) + 1.2))
    y = np.arange(len(labels))[::-1]
    for j, s in enumerate(styles):
        ax.scatter(rows[:, j], y, s=36, color=C[j], label=s, zorder=3, edgecolor=plots.SURFACE, linewidth=1.5)
    ax.axvline(0, color=plots.INK, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("median strategy XIRR minus Buy & Hold (pts/yr)   |   0 = Buy & Hold")
    ax.set_title("What matters: median result for each setting of each dial")
    ax.legend(loc="lower left")
    plots.note(fig, "Each dot = median over every configuration sharing that setting. Source: results/07_plateau/plateau.csv")
    plots.save(fig, FIG / "fig02_what_matters.png")

    # fig03 heatmaps
    lim = 0.4
    norm = plots.diverging_norm(lim)
    row_labels = [f"P{p} · {d}" for p in PIECES for d in DL_ORDER]
    for s in styles:
        for kind in ("fixed", "vol"):
            refs_ = ["high_5d", "high_10d", "high_20d", "high_50d"]
            res_ = ["A", "B", "C5", "C20"]
            fig, axes = plt.subplots(len(res_), len(refs_), figsize=(4.2 * len(refs_), 4.6 * len(res_)), squeeze=False)
            ths = THR[kind]
            for i, re_ in enumerate(res_):
                for j, rf in enumerate(refs_):
                    ax = axes[i, j]
                    sub = prim[(prim["style"] == s) & (prim["ref"] == rf) & (prim["reentry"] == re_) & (prim["kind"] == kind)]
                    grid = np.full((len(row_labels), len(ths)), np.nan)
                    for _, r in sub.iterrows():
                        rr = PIECES.index(int(r["pieces"])) * 4 + DL_ORDER.index(str(r["deadline"]))
                        cc = int(np.argmin([abs(t - r["thr"]) for t in ths]))
                        grid[rr, cc] = r["d_xirr_bh"] * 100
                        if r["qualifies"]:
                            ax.text(cc, rr, "Q", ha="center", va="center", fontsize=7, color=plots.INK, weight="bold")
                        elif r["d_xirr_bh"] > 0 and r["candidate"] and not np.isnan(r["share_bh"]):
                            ax.text(cc, rr, f"{r['share_bh']*100:.0f}", ha="center", va="center", fontsize=6, color=plots.INK)
                    im = ax.imshow(grid, cmap=plots.DIVERGING, norm=norm, aspect="auto")
                    cand_th = set(sub.loc[sub["candidate"], "thr"].round(6))
                    ax.set_xticks(range(len(ths)))
                    ax.set_xticklabels([(f"{t:.2%}" if kind == "fixed" else f"{t:g}x") + ("*" if round(t, 6) in cand_th else "")
                                        for t in ths], rotation=90, fontsize=6)
                    ax.set_yticks(range(len(row_labels)))
                    ax.set_yticklabels(row_labels if j == 0 else [], fontsize=6)
                    ax.grid(False)
                    ax.set_title(f"{rf} · {re_}", fontsize=9)
            cb = fig.colorbar(im, ax=axes, shrink=0.4, location="right", pad=0.01)
            cb.set_label("strategy XIRR minus Buy & Hold (pts/yr), clipped at ±0.4\nblue = better, red = worse")
            fig.subplots_adjust(top=0.95, hspace=0.32, wspace=0.08)
            fig.suptitle(f"{s} plateau map, {kind} drops: rows = pieces · deadline, columns = drop size (* = candidate rule)\n"
                         f"blue = beats Buy & Hold. Numbers = % of neighbours that also beat it (candidates only). Q = qualified",
                         fontsize=11, weight="bold", color=plots.INK, y=0.985)
            plots.save(fig, FIG / f"fig03_heatmap_{s}_{kind}.png")

    # fig04 break-even
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, s in zip(axes, styles):
        d = prim[prim["style"] == s]
        x, yv = (d["pred_d_bh"] * 100).clip(-2, 0.3), (d["d_xirr_bh"] * 100).clip(-2, 0.3)
        ax.scatter(x, yv, s=6, color=C[0], alpha=0.35, linewidth=0)
        ax.plot([-2, 0.3], [-2, 0.3], color=plots.INK2, linewidth=1)
        ax.text(-1.95, -1.75, "on the line = zero timing skill\n(result fully explained by waiting + fees)", fontsize=8, color=plots.INK2)
        ax.set_xlim(-2, 0.3)
        ax.set_ylim(-2, 0.3)
        ax.set_xlabel("predicted from wait & fees only (pts/yr)")
        ax.set_ylabel("actual (pts/yr)")
        ax.set_title(f"{s}: break-even formula vs reality (corr {d['d_xirr_bh'].corr(d['pred_d_bh']):.2f})")
    plots.note(fig, "predicted = -(B&H return x avg wait/365 + extra fees) / avg years invested. Values clipped to [-2, +0.3].")
    plots.save(fig, FIG / "fig04_breakeven.png")

    # fig05 over time vs Buy & Hold
    for s in styles:
        px, dist, start, end = _DATA[s]
        bh = run(px, dist, start, end, BuyHold()).daily["value"]
        lines = [("monthly DCA", MonthlyDCA(1)), ("perfect-timing ceiling (hindsight)", PerfectTimingCeiling())]
        for _, f in fin[fin["style"] == s].iterrows() if len(fin) else []:
            lines.append((f"{f['type']}: {f['rule']} P{int(f['pieces'])} D{f['deadline']}"
                          + ("" if f["label"] == "QUALIFIED" else " (not qualified)"), policy_of(s, f)))
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.axhline(0, color=plots.INK, linewidth=1)
        for k, (lab, pol) in enumerate(lines):
            v = run(px, dist, start, end, pol).daily["value"]
            rel = (v / bh - 1) * 100
            ax.plot(rel.index, rel.values, color=C[k + 1], linewidth=1.6)
            ax.text(rel.index[-1], rel.values[-1], f"  {lab}", color=plots.INK2, fontsize=8, va="center")
        ax.set_ylabel("account value vs Buy & Hold (%)")
        ax.set_title(f"{s}: account value relative to Buy & Hold over time (0 = same as Buy & Hold)")
        ax.set_xlim(bh.index[0], bh.index[-1] + (bh.index[-1] - bh.index[0]) * 0.35)
        import matplotlib.dates as mdates
        ax.set_xticks([t for t in ax.get_xticks() if t <= mdates.date2num(bh.index[-1])])
        ax.spines["bottom"].set_bounds(mdates.date2num(bh.index[0]), mdates.date2num(bh.index[-1]))
        plots.note(fig, "Same contributions for everyone. Above 0 = ahead of Buy & Hold that day. "
                        "The first year is noisy because the account is still tiny.")
        plots.save(fig, FIG / f"fig05_vs_buyhold_{s}.png")

    # fig06 bootstrap
    if boot_store:
        items = sorted(boot_store.items(), key=lambda kv: -np.mean(kv[1] > 0))[:6]
        fig, axes = plt.subplots(1, len(items), figsize=(3.6 * len(items), 3.4), squeeze=False)
        for ax, (idx, v) in zip(axes[0], items):
            r = prim.loc[idx]
            ax.hist(v * 100, bins=30, color=C[0], edgecolor=plots.SURFACE, linewidth=0.5)
            ax.axvline(0, color=plots.INK, linewidth=1)
            ax.set_title(f"{r['style']} {r['rule']}\nP{int(r['pieces'])} D{r['deadline']}: wins {np.mean(v > 0):.0%}", fontsize=8)
            ax.set_xlabel("vs B&H (pts/yr)")
        plots.note(fig, f"{BOOT_PATHS} alternative histories (63-day blocks of real days). Needs >= 70% wins to qualify.")
        plots.save(fig, FIG / "fig06_bootstrap.png")

    # fig07 PBO
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for ax, s in zip(axes, styles):
        p, lg = pbo[s]
        if not len(lg):
            ax.set_title(f"{s}: PBO not computed (fewer than 2 candidate configs)")
            continue
        ax.hist(lg, bins=20, color=C[0], edgecolor=plots.SURFACE, linewidth=0.5)
        ax.axvline(0, color=plots.INK, linewidth=1)
        ax.set_title(f"{s}: PBO = {p:.0%}")
        ax.set_xlabel("logit of out-of-sample rank of the in-sample winner  (< 0 = below median)")
    plots.note(fig, "CSCV, 8 blocks, 70 splits, candidate configs only. PBO near 50% = picking the in-sample best is a coin flip.")
    plots.save(fig, FIG / "fig07_pbo.png")


def write_finalists_md(fin, prim, pbo):
    lines = ["# Finalists for the holdout", "",
             f"Generated by scripts/7_plateau.py on {pd.Timestamp.now():%Y-%m-%d}, BEFORE the holdout is opened.",
             "Rules: notes/preregistration.md §9. Full table: results/07_plateau/plateau.csv", ""]
    for s in PERIOD:
        ps = prim[prim["style"] == s]
        lines.append(f"## {s}")
        lines.append(f"- qualified configurations: {int(ps['qualifies'].sum())}; PBO {pbo[s][0]:.0%}")
        fs = fin[fin["style"] == s] if len(fin) else fin
        if not len(fs):
            lines.append("- no finalist and no reference strategy (nothing beat Buy & Hold)")
        for _, f in fs.iterrows():
            lines.append(f"- **{f['label']}** type {f['type']}: `{f['rule']}` pieces {int(f['pieces'])}, deadline {f['deadline']}, "
                         f"fill {f['trade_through_bp']:g} bp | build d_xirr_bh {f['d_xirr_bh']*100:+.3f} pts/yr | "
                         f"neighbours {f['share_bh']:.0%} | bootstrap {f['boot_p'] if pd.notna(f['boot_p']) else 'n/a'}")
        lines.append("")
    (NOTES / "finalists.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
