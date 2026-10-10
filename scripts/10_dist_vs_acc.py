"""
10_dist_vs_acc.py  -  STUDY 07 (pre-registered §11): Distributing vs Accumulating fund.
                      POST-HOLDOUT, exploratory (design logged in notes/decisions.md first).

Same index, same fund family, same TER. The only difference is what happens to the dividends:
  Dist (IUSA)       pays them out -> 26% tax at once -> the rest waits for your next EUR 500
  Acc  (CSPX/SXR8)  reinvests them inside the fund -> no tax until you sell, then 26% on the gain

PART A  official period (Acc fund launched 2010-05-19 -> last day). Both funds priced from their
        OFFICIAL NAV in EUR on the same days, so the only difference is the dividend treatment.
        Quarterly B&H, monthly DCA and the E2 reference config, in each fund.
        Plus Dist priced from the IUSA market close (consistency with the rest of the project).
PART B  where the gap comes from (quarterly B&H):
          tax drag          = Dist without dividend tax  - Dist
          fund + reinvest   = Acc - Dist without dividend tax   (fund-level difference + the
                              dividends waiting in cash until your next contribution)
PART C  does the holding period matter? Rolling windows (a new start every quarter) of
        5 / 10 / 15 / 20 years, sold at the end. Before 2010-05-19 the Acc fund doesn't exist:
        its price is the Dist fund's official TOTAL RETURN index, spliced on at launch
        (marked "proxy" in the chart; A shows the two track each other).
Outputs: results/10_dist_vs_acc/*.csv, figures/fig15..fig17*.png, rows in results/master.csv
Run from project root:  python scripts/10_dist_vs_acc.py
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import plots                                                       # noqa: E402
from src.benchmarks import BuyHold, MonthlyDCA                             # noqa: E402
from src.costs import IBKR_TIERED, with_fractional                                           # noqa: E402
from src.data import PROC, load_post_holdout                                # noqa: E402
from src.engine import run                                                  # noqa: E402
from src.metrics import summarize, xirr                                     # noqa: E402
from src.signals import Rule                                                # noqa: E402
from src.strategy import DipStrategy                                        # noqa: E402

OUT = ROOT / "results" / "10_dist_vs_acc"
FIG = ROOT / "figures"
COST = IBKR_TIERED
NO_DIV_TAX = replace(IBKR_TIERED, name="ibkr_tiered_no_div_tax", div_tax=0.0)
HORIZONS = (5, 10, 15, 20)
NO_DIST = pd.DataFrame({"ex_date": pd.to_datetime([]), "pay_date": pd.to_datetime([]), "amount_eur": []})
pd.set_option("display.width", 220)


def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def as_bars(s):
    """A close-only series as the OHLC frame the engine expects (NAV = one price a day)."""
    s = s.dropna()
    return pd.DataFrame({"open": s, "high": s, "low": s, "close": s}, index=s.index)


def load_navs(px):
    fx = px["usd_per_eur"].dropna().sort_index()
    st = pd.read_csv(PROC / "ishares" / "Storico.csv", header=None, dtype=str)
    st.columns = st.iloc[0].str.strip()
    st = st.iloc[1:]
    d = pd.DataFrame({"date": pd.to_datetime(st["Al"], format="%d/%m/%Y", errors="coerce"),
                      "nav_usd": pd.to_numeric(st["NAV"], errors="coerce"),
                      "tr_usd": pd.to_numeric(st["Serie Fund Return"], errors="coerce")})
    d = d.dropna(subset=["date", "nav_usd"]).set_index("date").sort_index()
    d = d[~d.index.duplicated(keep="last")]
    f = fx.reindex(d.index, method="ffill")
    dist_nav, dist_tr = d["nav_usd"] / f, d["tr_usd"] / f
    acc = pd.read_parquet(PROC / "pit_acc_nav.parquet")["nav_eur"].dropna().sort_index()
    return dist_nav.dropna(), dist_tr.dropna(), acc


def reference_policy():
    p = ROOT / "results" / "07_plateau" / "finalists.csv"
    if not p.exists():
        return None
    f = pd.read_csv(p)
    f = f[f["style"] == "E2"]
    if not len(f):
        return None
    r = f.iloc[0]
    re_ = r["reentry"]
    rule = (Rule(r["ref"], r["kind"], float(r["thr"]), "C", cooldown=int(re_[1:])) if str(re_).startswith("C")
            else Rule(r["ref"], r["kind"], float(r["thr"]), re_))
    dl = int(r["deadline"]) if str(r["deadline"]).isdigit() else str(r["deadline"])
    return lambda: DipStrategy(rule, "E2", int(r["pieces"]), dl, 0.0), f"E2 reference {r['rule']} P{int(r['pieces'])} D{r['deadline']}"


def receivable(res, dist, cost):
    """Dividends already earned (ex-date passed) but not yet paid, net of tax, each day.
    The engine adds a dividend to cash on its PAY date, so between ex and pay dates the Dist
    account would look ~one dividend poorer. Harmless when every policy holds the same fund;
    NOT harmless when comparing Dist with Acc, so it's added back here."""
    d = res.daily
    out = pd.Series(0.0, index=d.index)
    for r in dist.itertuples():
        ex, pay = pd.Timestamp(r.ex_date), pd.Timestamp(r.pay_date)
        i = d.index.searchsorted(ex)
        if i == 0 or i >= len(d):
            continue
        held = d["shares"].iloc[i - 1]                    # shares owned the evening before the ex-date
        mask = (d.index >= d.index[i]) & (d.index < pay)
        out[mask] += held * float(r.amount_eur) * (1 - cost.div_tax)
    return out


def one(bars, dist, start, end, policy, cost=COST):
    res = run(bars, dist, start, end, policy, cost=cost)
    t = res.totals
    left = t["contributed"] + t["div_gross"] - t["div_tax"]
    right = t["cash_end"] + t["spent_on_shares"] + t["fees"]
    assert abs(left - right) < 1e-6, "accounting broken"
    m = summarize(res, cost)
    rec = receivable(res, dist, cost)
    res.daily["value_incl_receivable"] = res.daily["value"] + rec
    rec_end = float(rec.iloc[-1])
    gain = t["value_end"] - t["cash_end"] - t["cost_basis"]
    m["receivable_end"] = rec_end
    m["cgt_at_sale"] = max(gain, 0.0) * cost.cgt
    m["final_value"] = t["value_end"] + rec_end
    m["value_after_sell"] = m["final_value"] - m["cgt_at_sale"]
    m["total_tax"] = m["div_tax"] + m["cgt_at_sale"] + rec_end / (1 - cost.div_tax) * cost.div_tax
    m["xirr_before_sell"] = xirr(res.flows + [(t["end_date"], m["final_value"])])
    m["xirr_after_sell"] = xirr(res.flows + [(t["end_date"], m["value_after_sell"])])
    return m, res


def after_sell_path(res, cgt=0.26):
    """Account value each day IF you sold that day (26% on the unrealized gain)."""
    d = res.daily.copy()
    d["value"] = d["value_incl_receivable"]
    tr = res.trades
    basis = (tr.assign(c=tr["value"] + tr["fee"]).groupby("date")["c"].sum().cumsum()
             .reindex(d.index, method="ffill").fillna(0.0))
    gain = (d["value"] - d["cash"] - basis).clip(lower=0)
    return d["value"] - cgt * gain


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px, dist = load_post_holdout()
    dist_nav, dist_tr, acc = load_navs(px)

    # ============ PART A: official period, same days for both funds
    common = dist_nav.index.intersection(acc.index)
    start, end = common[0], common[-1]
    D, A = as_bars(dist_nav.loc[common]), as_bars(acc.loc[common])
    M = as_bars(px["close"])
    pols = [("quarterly B&H", lambda: BuyHold()), ("monthly DCA", lambda: MonthlyDCA(1))]
    ref = reference_policy()
    if ref:
        pols.append((ref[1], ref[0]))
    rows, runs = [], {}
    for name, make in pols:
        for fund, bars, dv, cost in [("Dist", D, dist, COST), ("Acc", A, NO_DIST, COST),
                                     ("Dist, no dividend tax", D, dist, NO_DIV_TAX),
                                     ("Dist (IUSA market close)", M, dist, COST),
                                     ("Dist, fractional shares", D, dist, with_fractional(COST)),
                                     ("Acc, fractional shares", A, NO_DIST, with_fractional(COST))]:
            m, res = one(bars, dv, start, end, make(), cost)
            m.update({"strategy": name, "fund": fund})
            rows.append(m)
            runs[(name, fund)] = res
    A_df = pd.DataFrame(rows)
    A_df["study"] = "07_dist_vs_acc"
    A_df.to_csv(OUT / "official_period.csv", index=False)

    # fund-level check: same days, NAV to NAV
    yrs = (end - start).days / 365.25
    cagr = lambda s: (s.asof(end) / s.asof(start)) ** (1 / yrs) - 1                      # noqa: E731

    # ============ PART C: rolling windows on the spliced series
    k = dist_tr.asof(acc.index[0])
    acc_long = pd.concat([dist_tr[dist_tr.index < acc.index[0]] * acc.iloc[0] / k, acc])
    long_days = dist_nav.index.intersection(acc_long.index)
    DL, AL = as_bars(dist_nav.loc[long_days]), as_bars(acc_long.loc[long_days])
    qstarts = pd.Series(long_days, index=long_days).groupby(long_days.to_period("Q")).min()
    roll = []
    for H in HORIZONS:
        for s0 in qstarts:
            e0 = s0 + pd.DateOffset(years=H)
            if e0 > long_days[-1]:
                break
            e0 = long_days[long_days <= e0][-1]
            md, _ = one(DL, dist, s0, e0, BuyHold())
            ma, _ = one(AL, NO_DIST, s0, e0, BuyHold())
            roll.append({"horizon": H, "start": s0.date(), "end": e0.date(), "uses_proxy": s0 < acc.index[0],
                         "gap_before_sell": ma["xirr_before_sell"] - md["xirr_before_sell"],
                         "gap_after_sell": ma["xirr_after_sell"] - md["xirr_after_sell"],
                         "eur_gap_after_sell": ma["value_after_sell"] - md["value_after_sell"],
                         "contributed": md["contributed"]})
    R = pd.DataFrame(roll)
    R.to_csv(OUT / "rolling_windows.csv", index=False)
    master = ROOT / "results" / "master.csv"
    old = pd.read_csv(master, low_memory=False) if master.exists() else pd.DataFrame()
    if len(old):
        old = old[old["study"] != "07_dist_vs_acc"]
    pd.concat([old, A_df], ignore_index=True).to_csv(master, index=False)

    # ============ PRINT
    section(f"STUDY 07 Dist vs Acc  (POST-HOLDOUT, exploratory)  {start.date()} -> {end.date()} ({yrs:.1f} yrs)")
    print("fund level, NAV to NAV in EUR, no investor taxes:")
    print(f"  Acc NAV {cagr(acc):.2%}/yr | Dist total return {cagr(dist_tr):.2%}/yr | Dist price {cagr(dist_nav):.2%}/yr")
    print(f"  -> Acc - Dist TR {100 * (cagr(acc) - cagr(dist_tr)):+.2f} pts/yr (same fund, ~0 expected)")
    print(f"  -> Dist price is {100 * (cagr(dist_tr) - cagr(dist_nav)):.2f} pts/yr below its TR = the dividends paid out")
    for name, _ in pols:
        print(f"\n{name}  (EUR 500/quarter, IBKR Tiered, whole shares)")
        t = A_df[A_df["strategy"] == name].set_index("fund")
        base = t.loc["Dist"]
        for fund, r in t.iterrows():
            print(f"  {fund:26s} XIRR before sell {r['xirr_before_sell']:6.2%} | after sell {r['xirr_after_sell']:6.2%} "
                  f"(vs Dist {100 * (r['xirr_after_sell'] - base['xirr_after_sell']):+.2f}) | "
                  f"final EUR {r['final_value']:,.0f} -> after sell {r['value_after_sell']:,.0f} | "
                  f"div tax {r['div_tax']:,.0f} + tax at sale {r['cgt_at_sale']:,.0f} = {r['total_tax']:,.0f} | "
                  f"fees {r['fees']:,.0f} | buys {r['n_buys']} | avg cash {r['avg_cash_pct']:.2%}")
    q = A_df[A_df["strategy"] == "quarterly B&H"].set_index("fund")
    section("PART B  where the gap comes from (quarterly B&H, pts/yr)")
    tax = q.loc["Dist, no dividend tax", "xirr_before_sell"] - q.loc["Dist", "xirr_before_sell"]
    fund = q.loc["Acc", "xirr_before_sell"] - q.loc["Dist, no dividend tax", "xirr_before_sell"]
    after = q.loc["Acc", "xirr_after_sell"] - q.loc["Dist", "xirr_after_sell"]
    print(f"  if you keep holding:    Acc {100 * (tax + fund):+.2f} = dividend tax drag {100 * tax:+.2f} "
          f"+ fund level & reinvestment delay {100 * fund:+.2f}")
    print(f"  if you sell at the end: Acc {100 * after:+.2f}  (smaller: the Dist investor's taxed dividends raised his cost")
    print("                          basis, so he pays less tax at the sale. What's left = the value of paying tax LATER)")
    fr = q.loc["Acc, fractional shares", "xirr_after_sell"] - q.loc["Dist, fractional shares", "xirr_after_sell"]
    frh = q.loc["Acc, fractional shares", "xirr_before_sell"] - q.loc["Dist, fractional shares", "xirr_before_sell"]
    print(f"  with fractional shares: Acc {100 * frh:+.2f} keep holding | {100 * fr:+.2f} sold at the end "
          f"(whole shares cost Acc {100 * (frh - (tax + fund)):+.2f} pts/yr: one Acc share costs about "
          f"EUR {A['close'].iloc[-1]:,.0f} today vs EUR {D['close'].iloc[-1]:,.0f} for Dist)")
    section("PART C  holding period: Acc minus Dist, XIRR after selling at the end (pts/yr)")
    for H, g in R.groupby("horizon"):
        a = g["gap_after_sell"] * 100
        b = g["gap_before_sell"] * 100
        print(f"  {H:2d} years: {len(g):3d} windows | median {a.median():+.2f} | worst {a.min():+.2f} | best {a.max():+.2f} | "
              f"Acc ahead in {(a > 0).mean():.0%} | median EUR gap {g['eur_gap_after_sell'].median():+,.0f} "
              f"on EUR {g['contributed'].median():,.0f} contributed | proxy windows {int(g['uses_proxy'].sum())} "
              f"| (keep holding: median {b.median():+.2f})")

    charts(runs, R, q, start, end)
    section("WROTE")
    print(f"  {OUT.relative_to(ROOT)}/official_period.csv, rolling_windows.csv")
    print("  figures/fig15_dist_vs_acc_over_time.png, fig16_dist_vs_acc_horizon.png, fig17_dist_vs_acc_gap.png")
    print("DONE. Paste everything back.")


def charts(runs, R, q, start, end):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    plots.style()
    C = plots.SERIES

    # fig15: Acc vs Dist account over time
    d, a = runs[("quarterly B&H", "Dist")], runs[("quarterly B&H", "Acc")]
    hold = (a.daily["value_incl_receivable"] / d.daily["value_incl_receivable"] - 1) * 100
    sold = (after_sell_path(a) / after_sell_path(d) - 1) * 100
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axhline(0, color=plots.INK, linewidth=1)
    ax.plot(hold.index, hold.values, color=C[0], linewidth=1.6)
    ax.plot(sold.index, sold.values, color=C[1], linewidth=1.6)
    gap = 0.06 * (max(hold.max(), sold.max()) - min(hold.min(), sold.min(), 0) + 1e-9)
    ys = [hold.iloc[-1], sold.iloc[-1]]
    if abs(ys[0] - ys[1]) < gap:
        ys = [max(ys) + gap / 2, min(ys) - gap / 2] if ys[0] >= ys[1] else [min(ys) - gap / 2, max(ys) + gap / 2]
    ax.text(hold.index[-1], ys[0], f"  if you keep holding: Acc {hold.iloc[-1]:+.1f}%", color=C[0], fontsize=8, va="center")
    ax.text(sold.index[-1], ys[1], f"  if you sold that day: Acc {sold.iloc[-1]:+.1f}%", color=C[1], fontsize=8, va="center")
    ax.set_xlim(hold.index[0], hold.index[-1] + (hold.index[-1] - hold.index[0]) * 0.35)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xticks([t for t in ax.get_xticks() if t <= mdates.date2num(hold.index[-1])])
    ax.spines["bottom"].set_bounds(mdates.date2num(hold.index[0]), mdates.date2num(hold.index[-1]))
    ax.set_ylabel("Acc account vs Dist account (%)")
    ax.set_title(f"{start.year}-{end.year}: how far ahead the Acc fund is (same EUR 500 a quarter, same days)")
    plots.note(fig, "Both funds priced from their official NAV in EUR. Dist: 26% tax on every dividend, the rest reinvested "
                    "with the next contribution. 'Sold that day' = 26% on the unrealized gain, for both.")
    plots.save(fig, FIG / "fig15_dist_vs_acc_over_time.png")

    # fig16: holding period
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    rng = np.random.default_rng(0)
    for ax, col, title in [(axes[0], "gap_before_sell", "if you keep holding"),
                           (axes[1], "gap_after_sell", "if you sell at the end")]:
        ax.axhline(0, color=plots.INK, linewidth=1)
        for j, H in enumerate(HORIZONS):
            g = R[R["horizon"] == H]
            if not len(g):
                continue
            x = j + rng.uniform(-0.18, 0.18, len(g))
            ax.scatter(x[~g["uses_proxy"].to_numpy()], (g[col] * 100)[~g["uses_proxy"]], s=12, color=C[0], linewidth=0)
            ax.scatter(x[g["uses_proxy"].to_numpy()], (g[col] * 100)[g["uses_proxy"]], s=12, color=plots.MUTED, linewidth=0)
            med = g[col].median() * 100
            ax.plot([j - 0.3, j + 0.3], [med, med], color=plots.INK, linewidth=2)
            ax.text(j + 0.32, med, f" {med:+.2f}", fontsize=8, va="center", color=plots.INK)
        ax.set_xticks(range(len(HORIZONS)))
        ax.set_xticklabels([f"{h} years" for h in HORIZONS])
        ax.set_xlim(-0.5, len(HORIZONS) - 0.3)
        ax.set_title(f"Acc minus Dist, {title}")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("XIRR difference (pts/yr)")
    plots.note(fig, "One dot per start quarter. Blue = official Acc NAV only. Grey = starts before 2010-05-19 (Acc priced from the "
                    "Dist fund's total-return index). Black bar = median.")
    plots.save(fig, FIG / "fig16_dist_vs_acc_horizon.png")

    # fig17: where the gap comes from
    tax = (q.loc["Dist, no dividend tax", "xirr_before_sell"] - q.loc["Dist", "xirr_before_sell"]) * 100
    fund = (q.loc["Acc", "xirr_before_sell"] - q.loc["Dist, no dividend tax", "xirr_before_sell"]) * 100
    after = (q.loc["Acc", "xirr_after_sell"] - q.loc["Dist", "xirr_after_sell"]) * 100
    items = [("keep holding: total Acc advantage", tax + fund, C[0]),
             ("   of which: 26% tax paid on every dividend", tax, C[0]),
             ("   of which: fund level + dividends waiting in cash", fund, C[0]),
             ("sell at the end: Acc advantage (paying tax later)", after, C[1])]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    y = np.arange(len(items))[::-1]
    v = [t[1] for t in items]
    ax.barh(y, v, color=[t[2] if t[1] > 0 else "#e34948" for t in items], height=0.55)
    ax.axvline(0, color=plots.INK, linewidth=1)
    for yi, x in zip(y, v):
        ax.text(x, yi, f" {x:+.2f}" if x >= 0 else f"{x:+.2f} ", va="center", ha="left" if x >= 0 else "right",
                fontsize=8, color=plots.INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([t[0] for t in items], fontsize=8)
    ax.grid(axis="y", visible=False)
    lo, hi = min(0, *v), max(0.1, *v)
    ax.set_xlim(lo - 0.15 * (hi - lo), hi + 0.25 * (hi - lo))
    ax.set_xlabel("XIRR points per year in favour of Acc")
    ax.set_title(f"{start.year}-{end.year}: where the Acc advantage comes from (quarterly B&H)")
    plots.note(fig, "Split computed with a third investor: Dist with NO tax on dividends. Selling shrinks the gap because the Dist "
                    "investor's already-taxed dividends raise his cost basis, so he pays less tax at the sale.")
    plots.save(fig, FIG / "fig17_dist_vs_acc_gap.png")


if __name__ == "__main__":
    main()
