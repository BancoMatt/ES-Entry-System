"""
11_crash_playbook.py  -  STUDY 11 (POST-HOLDOUT, exploratory; design in notes/decisions.md first)
"I invest EUR 500 every quarter. A crash comes. What should I do?"

Every line keeps quarterly B&H and adds ONE behaviour (src/crash.py). Scored AFTER SELLING at
the end (taxes realised along the way, e.g. by panic selling, count), vs plain quarterly B&H on
the same period. Periods: 1999-2026 and three separate starts (1999-2008, 2009-2018, 2019-2026).

  A  behaviour      pause buying at -20% (restart at -10% / new high) | panic-sell at -20%
                    (buy back at -10% / new high)
  B  front-loading  at -X% invest the next N quarters now, borrowed from the emergency fund,
                    repaid by the next contributions. X 10..40%, N 1/2/4/8, one-shot or ladder.
                    Emergency fund in plain cash (0%) or in a deposit paying the ECB rate
                    (then the interest it would have earned is a cost).
  C  placebo        each front-load config (1999-2026, 0%) re-run with the SAME number of
                    front-loads on random days, 200x. Did the crash signal beat random timing?
  R  reserve        control line: 10% of each contribution kept as cash, invested at -20%.
  D  credit line    a big investor (EUR 1M lump sum, no salary). At -X% borrow f x portfolio
                    (ECB rate + 1.5%), buy; repay at the next high by selling, or carry to the end.
                    Margin call if the loan exceeds 50% of the portfolio.
Representative configs (fixed in advance, used for the per-crash tables and charts):
  front-load -20%, N = 4 quarters, one-shot, 0% | credit line -20%, borrow 20%
Outputs: results/11_crash_playbook/*.csv, figures/fig18..fig21*.png
Run from project root:  python scripts/11_crash_playbook.py      (PLACEBO=20 for a quick test)
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
from src.benchmarks import BuyHold                                          # noqa: E402
from src.costs import IBKR_TIERED                                           # noqa: E402
from src.crash import CreditLine, FrontLoad, Panic, Pause, Reserve, drawdown  # noqa: E402
from src.data import load_post_holdout                                      # noqa: E402
from src.engine import Engine                                               # noqa: E402
from src.metrics import summarize                                           # noqa: E402

OUT = ROOT / "results" / "11_crash_playbook"
FIG = ROOT / "figures"
COST = IBKR_TIERED
PERIODS = {"1999-2026": ("1999-01-04", None), "1999-2008": ("1999-01-04", "2008-12-31"),
           "2009-2018": ("2009-01-01", "2018-12-31"), "2019-2026": ("2019-01-01", None)}
TRIGGERS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
QUARTERS = (1, 2, 4, 8)
MODES = ("one", "ladder")
REP_FL = (0.20, 4, "one")
CL_TRIG, CL_FRAC, CL_REPAY = (0.15, 0.20, 0.30), (0.10, 0.20), ("recovery", "end")
REP_CL = (0.20, 0.20)
LUMP = 1_000_000.0
PLACEBO = int(os.environ.get("PLACEBO", 200))
SEED = 2026
WORKERS = max(1, (os.cpu_count() or 2) - 1)
pd.set_option("display.width", 220)

_D = {}


def _init(data):
    sys.path.insert(0, str(ROOT))
    _D["px"], _D["dist"], _D["rates"] = data


def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def bounds(period):
    a, b = PERIODS[period]
    return pd.Timestamp(a), (pd.Timestamp(b) if b else _D["px"].index[-1])


def _run(period, policy, rate=None, lump=False):
    start, end = bounds(period)
    eng = Engine(_D["px"], _D["dist"], start, end, policy, cost=COST,
                 debt_rate=_D["rates"][rate] if rate else None)
    if lump:
        eng.contrib = {eng.dates[0]: LUMP}
    res = eng.run()
    t = res.totals
    assert abs(t["books_residual"]) < 1e-6 * max(1.0, t["contributed"]), f"books broken: {policy.name}"
    m = summarize(res, COST)
    m.update({"period": period, "policy": policy.name, "borrowed": t["borrowed"], "interest": t["interest"],
              "tax_on_sales": t["cgt_paid"], "debt_end": t["debt_end"]})
    return m, res


def _job(job):
    kind = job[0]
    if kind == "base":
        m, _ = _run(job[1], BuyHold())
        return {**m, "part": "base"}
    if kind == "A":
        _, period, pol = job
        m, _ = _run(period, pol)
        return {**m, "part": "A", "events": getattr(pol, "events", [])}
    if kind == "R":
        m, _ = _run(job[1], Reserve())
        return {**m, "part": "R"}
    if kind == "B":
        _, period, X, N, mode, rate = job
        pol = FrontLoad(X, N, mode)
        m, _ = _run(period, pol, rate)
        return {**m, "part": "B", "trigger": X, "quarters": N, "mode": mode, "ef": rate or "cash 0%",
                "n_fires": len(pol.fires), "fires": pol.fires}
    if kind == "P":
        _, X, N, mode, days = job
        m, _ = _run("1999-2026", FrontLoad(X, N, mode, fire_days=days))
        return {"trigger": X, "quarters": N, "mode": mode, "xirr_after_sell": m["xirr_after_sell"]}
    if kind == "D":
        _, period, X, f, repay = job
        pol = CreditLine(X, f, repay)
        m, res = _run(period, pol, "credit", lump=True)
        out = {**m, "part": "D", "trigger": X, "frac": f, "repay": repay, "draws": pol.draws,
               "margin_calls": pol.margin_calls, "max_ltv": max(pol.ltv) if pol.ltv else 0.0}
        if period == "1999-2026" and (X, f) == REP_CL:
            out["daily"] = (res.daily["value"] - res.daily["debt"], pd.Series(pol.ltv, index=res.daily.index))
        return out
    if kind == "D0":
        pol = CreditLine(0.20, 0.0, "end")
        m, res = _run(job[1], pol, "credit", lump=True)
        out = {**m, "part": "D0"}
        if job[1] == "1999-2026":
            out["daily"] = (res.daily["value"] - res.daily["debt"], None)
        return out
    raise ValueError(kind)


def pmap(jobs, data, label, chunksize=4):
    t0, out = time.time(), []
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init, initargs=(data,)) as ex:
        for i, r in enumerate(ex.map(_job, jobs, chunksize=chunksize), 1):
            out.append(r)
            if i % max(1, len(jobs) // 5) == 0 or i == len(jobs):
                print(f"  {label}: {i:,}/{len(jobs):,}  ({time.time() - t0:.0f}s)")
    return out


def load_rates():
    f = ROOT / "data" / "raw" / "macro" / "ecb_dfr.csv"
    if not f.exists():
        sys.exit("missing data/raw/macro/ecb_dfr.csv: run  python scripts/0_fetch_data.py --only ecb_dfr")
    r = pd.read_csv(f, parse_dates=["date"]).set_index("date")["dfr_pct"].sort_index() / 100
    return {"deposit": r.clip(lower=0.0), "credit": (r + 0.015).clip(lower=0.0), "raw": r}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px, dist = load_post_holdout()
    rates = load_rates()
    data = (px, dist, rates)
    _init(data)

    # ---------------- all the runs except the placebo
    jobs = [("base", p) for p in PERIODS] + [("R", p) for p in PERIODS]
    for p in PERIODS:
        jobs += [("A", p, Pause(0.20, 0.10)), ("A", p, Pause(0.20, 0.0)),
                 ("A", p, Panic(0.20, 0.10)), ("A", p, Panic(0.20, 0.0))]
        jobs += [("B", p, X, N, m, r) for X in TRIGGERS for N in QUARTERS for m in MODES for r in (None, "deposit")]
        jobs += [("D0", p)] + [("D", p, X, f, rp) for X in CL_TRIG for f in CL_FRAC for rp in CL_REPAY]
    print(f"running {len(jobs):,} configurations on {WORKERS} cores")
    rows = pmap(jobs, data, "main runs")
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("fires", "draws", "margin_calls", "daily", "events")}
                       for r in rows])
    base = df[df["part"] == "base"].set_index("period")
    base_d = df[df["part"] == "D0"].set_index("period")
    for p in PERIODS:
        m = (df["period"] == p) & df["part"].isin(["A", "R", "B", "base"])
        md = (df["period"] == p) & df["part"].isin(["D", "D0"])
        for col in ("xirr_after_sell", "xirr_before_sell"):
            df.loc[m, "d_" + col] = df.loc[m, col] - base.loc[p, col]
            df.loc[md, "d_" + col] = df.loc[md, col] - base_d.loc[p, col]

    # ---------------- placebo: same number of front-loads, random days, 1999-2026, 0%
    full = df[(df["part"] == "B") & (df["period"] == "1999-2026") & (df["ef"] == "cash 0%")]
    s, e = pd.Timestamp(PERIODS["1999-2026"][0]), px.index[-1]
    n_days = int(((px.index >= s) & (px.index <= e)).sum())
    rng = np.random.default_rng(SEED)
    pjobs = []
    for _, r in full.iterrows():
        k = int(r["n_fires"])
        for _ in range(PLACEBO if k > 0 else 0):
            pjobs.append(("P", r["trigger"], r["quarters"], r["mode"],
                          sorted(rng.choice(n_days - 1, size=k, replace=False).tolist())))
    print(f"placebo: {len(pjobs):,} random-date runs")
    plac = pd.DataFrame(pmap(pjobs, data, "placebo", chunksize=16)) if pjobs else pd.DataFrame()
    if len(plac):
        plac["d_xirr_after_sell"] = plac["xirr_after_sell"] - base.loc["1999-2026", "xirr_after_sell"]
        key = ["trigger", "quarters", "mode"]
        real = full.set_index(key)["d_xirr_after_sell"]
        g = plac.groupby(key)["d_xirr_after_sell"]
        pct = g.apply(lambda v: np.mean(v.to_numpy() < real.loc[v.name]))
        df = df.merge(pct.rename("placebo_pct").reset_index().assign(period="1999-2026", ef="cash 0%", part="B"),
                      on=["part", "period", "ef"] + key, how="left")
        df = df.merge(g.median().rename("placebo_median").reset_index().assign(period="1999-2026", ef="cash 0%", part="B"),
                      on=["part", "period", "ef"] + key, how="left")
        plac.to_csv(OUT / "placebo.csv", index=False)
    df.to_csv(OUT / "runs.csv", index=False)

    # ---------------- per-crash tables (representative configs)
    closes = px["close"]
    rep = next(r for r in rows if r["part"] == "B" and r["period"] == "1999-2026" and r["ef"] == "cash 0%"
               and (r["trigger"], r["quarters"], r["mode"]) == REP_FL)
    contrib_days = pd.Series(px.index[(px.index >= s)]).groupby(px.index[(px.index >= s)].to_period("Q")).min().tolist()
    fl_rows = []
    for d0, dd0, amt in rep["fires"]:
        nq = int(np.ceil(amt / 500 - 1e-9))
        later = [d for d in contrib_days if d > d0][:nq]
        if not later:
            continue
        p0, pl = closes.loc[d0], closes.loc[later].mean()
        fl_rows.append({"date": d0.date(), "drawdown": dd0, "eur": amt, "paid_vs_scheduled_buys": p0 / pl - 1,
                        "eur_gain": amt * (pl / p0 - 1), "repaid_by": later[-1].date()})
    FL = pd.DataFrame(fl_rows)
    FL.to_csv(OUT / "frontload_per_crash.csv", index=False)
    dd_all = closes / closes.cummax() - 1
    cl_rows = []
    for rp in CL_REPAY:
        r = next(x for x in rows if x["part"] == "D" and x["period"] == "1999-2026" and (x["trigger"], x["frac"]) == REP_CL
                 and x["repay"] == rp)
        for d0, dd0, amt in r["draws"]:
            after = dd_all[dd_all.index > d0]
            hi = after[after >= -1e-12]
            d1 = hi.index[0] if (rp == "recovery" and len(hi)) else px.index[-1]
            rr = rates["credit"].reindex(px.loc[d0:d1].index, method="ffill").fillna(0)
            interest = amt * float((rr * px.loc[d0:d1].index.to_series().diff().dt.days.fillna(0) / 365).sum())
            ret = closes.loc[d1] / closes.loc[d0] - 1
            cl_rows.append({"repay": rp, "draw_date": d0.date(), "drawdown": dd0, "eur": amt, "until": d1.date(),
                            "years": (d1 - d0).days / 365.25, "price_gain": ret, "interest_eur": interest,
                            "net_before_tax_eur": amt * ret - interest})
        cl_rows += [{"repay": rp, "draw_date": "MARGIN CALL", "until": d.date(), "drawdown": ltv, "eur": need}
                    for d, ltv, need in r["margin_calls"]]
    CL = pd.DataFrame(cl_rows)
    CL.to_csv(OUT / "credit_line_per_draw.csv", index=False)

    # ---------------- print
    pts = lambda x: f"{100 * x:+.2f}"                                    # noqa: E731
    section("STUDY 11 crash playbook (POST-HOLDOUT, exploratory). vs quarterly B&H, XIRR AFTER selling, pts/yr")
    print("period         " + "".join(f"{p:>12s}" for p in PERIODS))
    print("B&H XIRR       " + "".join(f"{base.loc[p, 'xirr_after_sell']:>12.2%}" for p in PERIODS))
    for name in ["pause_-20%_until_-10%", "pause_-20%_until_new_high", "panic_sell_-20%_rebuy_-10%",
                 "panic_sell_-20%_rebuy_new_high", "reserve_10%_deploy_-20%"]:
        t = df[df["policy"] == name].set_index("period")
        print(f"{name:31s}" + "".join(f"{pts(t.loc[p, 'd_xirr_after_sell']):>12s}" if p in t.index else " " * 12 for p in PERIODS))
    for ef in ("cash 0%", "deposit"):
        t = df[(df["part"] == "B") & (df["ef"] == ef) & (df["trigger"] == REP_FL[0]) & (df["quarters"] == REP_FL[1])
               & (df["mode"] == REP_FL[2])].set_index("period")
        print(f"front-load -20% N4 ({ef:7s})   " + "".join(f"{pts(t.loc[p, 'd_xirr_after_sell']):>12s}" for p in PERIODS))
    section("B  FRONT-LOADING, 1999-2026, emergency fund in plain cash: vs B&H (pts/yr) | placebo percentile")
    for mode in MODES:
        print(f"\n{mode}-shot" if mode == "one" else "\nladder (another N at every further -10%)")
        print("trigger  " + "".join(f"{'N=' + str(n):>18s}" for n in QUARTERS))
        t = df[(df["part"] == "B") & (df["period"] == "1999-2026") & (df["ef"] == "cash 0%") & (df["mode"] == mode)]
        for X in TRIGGERS:
            cells = []
            for n in QUARTERS:
                r = t[(t["trigger"] == X) & (t["quarters"] == n)].iloc[0]
                pp = f" p{100 * r['placebo_pct']:.0f}" if "placebo_pct" in r and pd.notna(r.get("placebo_pct")) else ""
                cells.append(f"{pts(r['d_xirr_after_sell'])} ({int(r['n_fires'])}x){pp}")
            print(f"  -{X:.0%}   " + "".join(f"{c:>18s}" for c in cells))
    b = df[(df["part"] == "B") & (df["period"] == "1999-2026")]
    print(f"\nemergency fund in a deposit (ECB rate, forgone interest counted) vs plain cash, median over configs: "
          f"{pts(b[b['ef'] == 'deposit']['d_xirr_after_sell'].median())} vs {pts(b[b['ef'] == 'cash 0%']['d_xirr_after_sell'].median())}")
    if "placebo_pct" in df:
        pp = df["placebo_pct"].dropna()
        print(f"placebo percentile over {len(pp)} configs: median p{100 * pp.median():.0f} | "
              f"configs beating 90% of random dates: {(pp >= 0.9).sum()}")
    print("\nper crash, representative config (-20%, N=4, one-shot):  bought at vs the scheduled buys it replaced")
    for _, r in FL.iterrows():
        print(f"  {r['date']}  drawdown {r['drawdown']:6.1%}  EUR {r['eur']:,.0f}  paid {r['paid_vs_scheduled_buys']:+6.1%} "
              f"vs scheduled -> {r['eur_gain']:+,.0f} EUR  (repaid by {r['repaid_by']})")
    section("D  CREDIT LINE, big investor (EUR 1M lump sum), ECB rate + 1.5%: vs no loan, XIRR after selling")
    print("config                         " + "".join(f"{p:>12s}" for p in PERIODS) + "   margin calls (1999-2026) | max loan/portfolio")
    dd_ = df[df["part"] == "D"]
    for X in CL_TRIG:
        for f in CL_FRAC:
            for rp in CL_REPAY:
                t = dd_[(dd_["trigger"] == X) & (dd_["frac"] == f) & (dd_["repay"] == rp)].set_index("period")
                r = next(x for x in rows if x["part"] == "D" and x["period"] == "1999-2026" and x["trigger"] == X
                         and x["frac"] == f and x["repay"] == rp)
                print(f"  -{X:.0%} borrow {f:.0%} {rp:9s}         " + "".join(f"{pts(t.loc[p, 'd_xirr_after_sell']):>12s}" for p in PERIODS)
                      + f"   {len(r['margin_calls']):>3d} | {r['max_ltv']:.0%}")
    print("\nper draw, representative (-20%, borrow 20%):")
    for _, r in CL.iterrows():
        if r["draw_date"] == "MARGIN CALL":
            print(f"  [{r['repay']}] MARGIN CALL {r['until']}: loan {r['drawdown']:.0%} of portfolio, forced sale EUR {r['eur']:,.0f}")
        else:
            print(f"  [{r['repay']}] {r['draw_date']} at {r['drawdown']:6.1%}: EUR {r['eur']:,.0f} until {r['until']} "
                  f"({r['years']:.1f}y) | price {r['price_gain']:+.0%} | interest EUR {r['interest_eur']:,.0f} | "
                  f"net before tax EUR {r['net_before_tax_eur']:+,.0f}")
    charts(df, plac, rows, FL, rates)
    section("WROTE")
    print(f"  {OUT.relative_to(ROOT)}/runs.csv, placebo.csv, frontload_per_crash.csv, credit_line_per_draw.csv")
    print("  figures/fig18_crash_behaviour.png, fig19_frontload_map.png, fig20_frontload_placebo.png, fig21_credit_line.png")
    print("DONE. Paste everything back.")


def charts(df, plac, rows, FL, rates):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    plots.style()
    C = plots.SERIES
    better, worse = "#2a78d6", "#e34948"

    # fig18 behaviour, 4 periods
    items = [("pause at -20%, restart at -10%", "pause_-20%_until_-10%"),
             ("pause at -20%, restart at new high", "pause_-20%_until_new_high"),
             ("panic-sell at -20%, rebuy at -10%", "panic_sell_-20%_rebuy_-10%"),
             ("panic-sell at -20%, rebuy at new high", "panic_sell_-20%_rebuy_new_high"),
             ("reserve 10%, invest at -20%", "reserve_10%_deploy_-20%"),
             ("front-load 4 quarters at -20% (cash EF)", f"frontload_-20%_N4_one")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, p in zip(axes, PERIODS):
        v = []
        for lab, pol in items:
            t = df[(df["policy"] == pol) & (df["period"] == p)]
            if pol.startswith("frontload"):
                t = t[t["ef"] == "cash 0%"]
            v.append(t["d_xirr_after_sell"].iloc[0] * 100 if len(t) else np.nan)
        y = np.arange(len(items))[::-1]
        ax.barh(y, v, color=[better if x > 0 else worse for x in v], height=0.6)
        ax.axvline(0, color=plots.INK, linewidth=1)
        lim = np.nanmax(np.abs(v)) * 1.5 + 0.05
        ax.set_xlim(-lim, lim)
        for yi, x in zip(y, v):
            ax.text(x, yi, f" {x:+.2f}" if x >= 0 else f"{x:+.2f} ", va="center", ha="left" if x >= 0 else "right",
                    fontsize=7.5, color=plots.INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([t[0] for t in items], fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.set_title(p)
        ax.set_xlabel("vs quarterly B&H (pts/yr)")
    fig.suptitle("A crash comes: what each reaction did to your return (after selling at the end)", fontsize=11,
                 weight="bold", color=plots.INK)
    fig.tight_layout()
    plots.note(fig, "EUR 500 a quarter in IUSA, IBKR Tiered. Each period is a separate start. Scales differ per panel.")
    plots.save(fig, FIG / "fig18_crash_behaviour.png")

    # fig19 front-load map
    t = df[(df["part"] == "B") & (df["period"] == "1999-2026") & (df["ef"] == "cash 0%")]
    vmax = max(0.1, np.nanmax(np.abs(t["d_xirr_after_sell"] * 100)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, mode in zip(axes, MODES):
        g = t[t["mode"] == mode].pivot(index="trigger", columns="quarters", values="d_xirr_after_sell") * 100
        pp = t[t["mode"] == mode].pivot(index="trigger", columns="quarters", values="placebo_pct") if "placebo_pct" in t else None
        im = ax.imshow(g.values, cmap=plots.DIVERGING, norm=plots.diverging_norm(vmax), aspect="auto")
        for a in range(g.shape[0]):
            for b in range(g.shape[1]):
                txt = f"{g.values[a, b]:+.2f}"
                if pp is not None and pd.notna(pp.values[a, b]):
                    txt += f"\np{100 * pp.values[a, b]:.0f}"
                ax.text(b, a, txt, ha="center", va="center", fontsize=8, color=plots.INK)
        ax.set_xticks(range(g.shape[1]))
        ax.set_xticklabels([f"{int(n)} quarters" for n in g.columns])
        ax.set_yticks(range(g.shape[0]))
        ax.set_yticklabels([f"-{x:.0%}" for x in g.index])
        ax.set_ylabel("trigger: market this far below its high")
        ax.grid(False)
        ax.set_title("one-shot per crash" if mode == "one" else "ladder: again at every further -10%")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("vs quarterly B&H (pts/yr)")
    fig.suptitle("1999-2026: front-loading future contributions in a crash. pNN = beats NN% of random-date front-loads",
                 fontsize=11, weight="bold", color=plots.INK)
    plots.note(fig, "Emergency fund in plain cash (no interest lost). Blue = beats B&H. p50 = no better than front-loading on random days.")
    plots.save(fig, FIG / "fig19_frontload_map.png")

    # fig20 placebo for the representative config
    if len(plac):
        X, N, mode = REP_FL
        v = plac[(plac["trigger"] == X) & (plac["quarters"] == N) & (plac["mode"] == mode)]["d_xirr_after_sell"] * 100
        real = t[(t["trigger"] == X) & (t["quarters"] == N) & (t["mode"] == mode)]["d_xirr_after_sell"].iloc[0] * 100
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.hist(v, bins=30, color=plots.MUTED, edgecolor=plots.SURFACE, linewidth=0.5)
        ax.axvline(0, color=plots.INK, linewidth=1)
        ax.axvline(real, color=C[0], linewidth=2)
        top = ax.get_ylim()[1]
        ax.text(real, top * 0.95, f" crash-timed: {real:+.2f}\n beats {np.mean(v < real):.0%} of random dates",
                color=C[0], fontsize=8, va="top")
        ax.text(0, top * 0.7, " quarterly B&H", color=plots.INK, fontsize=8)
        ax.set_xlabel("vs quarterly B&H (pts/yr, after selling)")
        ax.set_ylabel("random-date runs")
        ax.set_title(f"Is it the crash, or just investing earlier? Front-load {N} quarters at -{X:.0%} vs the same on random days")
        plots.note(fig, f"Grey = {len(v)} runs with the same number of front-loads on random days, 1999-2026.")
        plots.save(fig, FIG / "fig20_frontload_placebo.png")

    # fig21 credit line over time
    r0 = next(x for x in rows if x["part"] == "D0" and x["period"] == "1999-2026")["daily"][0]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.axhline(0, color=plots.INK, linewidth=1)
    for j, rp in enumerate(CL_REPAY):
        r = next(x for x in rows if x["part"] == "D" and x["period"] == "1999-2026" and (x["trigger"], x["frac"]) == REP_CL
                 and x["repay"] == rp)
        net, ltv = r["daily"]
        rel = (net / r0 - 1) * 100
        col = C[0] if rp == "recovery" else C[1]
        lab = "repay at next high (sell, taxed)" if rp == "recovery" else "carry the loan to the end"
        a1.plot(rel.index, rel.values, color=col, linewidth=1.5)
        a1.text(rel.index[-1], rel.values[-1], f"  {lab}: {rel.values[-1]:+.1f}%", color=col, fontsize=8, va="center")
        a2.plot(ltv.index, ltv.values * 100, color=col, linewidth=1.3)
        for d0, _, _ in r["draws"]:
            a1.axvline(d0, color=col, linewidth=0.6, alpha=0.5)
    a2.axhline(50, color=worse, linewidth=1)
    a2.text(r0.index[0], 51, " margin call level (50%)", color=worse, fontsize=7.5, va="bottom")
    a1.set_ylabel("net worth vs no loan (%)")
    a2.set_ylabel("loan / portfolio (%)")
    a1.set_title("1999-2026, EUR 1M investor: borrow 20% of the portfolio at -20%, at the ECB rate + 1.5%")
    a1.set_xlim(r0.index[0], r0.index[-1] + (r0.index[-1] - r0.index[0]) * 0.35)
    a2.xaxis.set_major_locator(mdates.YearLocator(3))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    a2.set_xticks([x for x in a2.get_xticks() if x <= mdates.date2num(r0.index[-1])])
    plots.note(fig, "Thin vertical lines = the days the loan was drawn. Net worth = portfolio minus the loan. "
                    "Interest accrues on the loan; lender forces a sale back to 40% if the loan passes 50%.")
    plots.save(fig, FIG / "fig21_credit_line.png")


if __name__ == "__main__":
    main()
