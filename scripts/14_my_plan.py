"""
14_my_plan.py  -  STUDY 13: Matt's own plan (POST-HOLDOUT, personal planning, exploratory).

The question: with contributions that start small and grow (EUR 500/q -> 1,200/q -> 3,000/q
growing 5% a year), is 1.25x-1.5x leverage worth it, and HOW should it be held:
a loan on top of one fund ('margin') or a 1x fund + a 2x fund with no loan ('etf' split)?

PART A  Shiller monthly S&P 500 total return, every 30-year window starting each quarter
        1871 -> 1996 (~500 overlapping windows). USD, nominal. Borrowing = 10y yield + 1.5%
        (loan) or + 0.5% (inside the 2x fund) - see decisions.md for why that's a proxy.
        + the same with every borrowing rate 2 points higher (the "what if rates are high" check)
        + the same with a flat EUR 500/q (the study's usual investor) for comparison.
PART B  Real IUSA in EUR, daily, ECB deposit rate, starts 2000, Oct 2007 (market top), 2010,
        2020 -> last day. Same plan, same strategies.
Strategies (src/plan.py): 1x | loan 1.25x | loan 1.5x | loan 1.5x->1.25x after 5y |
                          ETF 75/25 | ETF 50/50 | ETF 50/50 -> 75/25 after 5y
Outputs: results/14_my_plan/*.csv, figures/fig28_my_plan_1871.png, fig29_my_plan_real.png
Run from project root:  python scripts/14_my_plan.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import plots                                           # noqa: E402
from src.data import PROC, load_post_holdout                    # noqa: E402
from src.metrics import xirr                                    # noqa: E402
from src.plan import const, drawdown_stats, glide, plan_flat, plan_matt, simulate   # noqa: E402

OUT = ROOT / "results" / "14_my_plan"
FIG = ROOT / "figures"
YEARS = 30
LOAN_SPREAD, FUND_SPREAD, FUND_TER = 0.015, 0.005, 0.010   # 2x fund: 0.6% TER + ~0.4% measured extra tracking gap
STRATS = [("1x (plain B&H)", "margin", const(1.0)),
          ("loan 1.25x", "margin", const(1.25)),
          ("loan 1.5x", "margin", const(1.5)),
          ("loan 1.5x -> 1.25x", "margin", glide),
          ("ETF 75/25 (1.25x)", "etf", const(1.25)),
          ("ETF 50/50 (1.5x)", "etf", const(1.5)),
          ("ETF 50/50 -> 75/25", "etf", glide)]
REAL_STARTS = {"2000": "2000-01-01", "Oct 2007 (top)": "2007-10-01", "2010": "2010-01-01", "2020": "2020-01-01"}
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 30)


# ------------------------------------------------------------------ PART A: Shiller windows
def shiller_inputs(extra_rate=0.0):
    sh = pd.read_parquet(PROC / "shiller_monthly.parquet").sort_index()
    P, D = sh["P"].astype(float), sh["D"].astype(float).ffill()
    r1 = ((P + D / 12) / P.shift(1) - 1).fillna(0.0)
    y = (sh["GS10"].astype(float).ffill() / 100).shift(1).bfill()    # last month's yield: known in advance
    base = y + extra_rate
    r2 = 2 * r1 - (base + FUND_SPREAD) / 12 - FUND_TER / 12
    mrate = (base + LOAN_SPREAD).clip(lower=0.0) / 12
    return sh.index, r1.to_numpy(), r2.to_numpy(), mrate.to_numpy()


def shiller_windows(plan, extra_rate=0.0, with_xirr=True):
    dates, r1, r2, mr = shiller_inputs(extra_rate)
    T = YEARS * 12
    starts = [i for i, d in enumerate(dates) if d.month in (1, 4, 7, 10) and i + T <= len(dates)]
    S = np.array(starts)
    idx = S[:, None] + np.arange(T)[None, :]
    R1, R2, MR = r1[idx], r2[idx], mr[idx]
    R1[:, 0] = R2[:, 0] = MR[:, 0] = 0.0                               # first buy at the start month's price
    k = np.arange(T) // 3
    C = np.where(np.arange(T) % 3 == 0, plan(k), 0.0)[None, :].repeat(len(S), 0)
    rows, ratios = [], {}
    base_after = None
    for name, mode, lev in STRATS:
        tg = lev(k)[None, :].repeat(len(S), 0)
        res = simulate(R1, R2, MR, C, tg, mode)
        mdd, under = drawdown_stats(res["nw_pre"], res["nw_post"], 12)
        if base_after is None:
            base_after = res["final_after"]
        ratio = res["final_after"] / base_after
        ratios[name] = ratio
        avg_expo = np.nanmean(res["exposure"][:, 1:], axis=1)
        for w, s in enumerate(S):
            row = {"strategy": name, "start": dates[s], "final_after_sell": res["final_after"][w],
                   "final_before_sell": res["final_before"][w], "x_vs_1x": ratio[w],
                   "margin_call": not res["alive"][w],
                   "call_date": dates[s + res["bust_step"][w]] if not res["alive"][w] else None,
                   "max_drawdown": mdd[w], "longest_underwater_years": under[w],
                   "avg_exposure": avg_expo[w], "interest": res["interest"][w],
                   "tax_on_rebalancing": res["tax_on_rebalancing"][w], "contributed": res["contributed"][w]}
            if with_xirr:
                fl = [(dates[s + t], -C[w, t]) for t in range(T) if C[w, t] > 0]
                row["xirr_after_sell"] = xirr(fl + [(dates[s + T - 1], res["final_after"][w])]) if res["alive"][w] else -1.0
            rows.append(row)
    return pd.DataFrame(rows), dates[S], ratios


def summary(df):
    out = []
    for name, _, _ in STRATS:
        t = df[df["strategy"] == name]
        worst = t.loc[t["x_vs_1x"].idxmin()]
        r = {"strategy": name,
             "median x vs 1x": t["x_vs_1x"].median(), "10th pct": t["x_vs_1x"].quantile(0.10),
             "worst x": worst["x_vs_1x"], "worst start": f"{worst['start']:%Y-%m}",
             "% behind 1x": (t["x_vs_1x"] < 1).mean() * 100, "margin calls": int(t["margin_call"].sum()),
             "median maxDD": t["max_drawdown"].median(), "worst maxDD": t["max_drawdown"].min(),
             "median underwater y": t["longest_underwater_years"].median(),
             "worst underwater y": t["longest_underwater_years"].max(),
             "avg exposure": t["avg_exposure"].median()}
        if "xirr_after_sell" in t:
            r["median XIRR after sell"] = t["xirr_after_sell"].median()
            r["10th pct XIRR"] = t["xirr_after_sell"].quantile(0.10)
        out.append(r)
    return pd.DataFrame(out)


def fmt(s):
    f = s.copy()
    for c in f.columns:
        if c in ("median x vs 1x", "10th pct", "worst x"):
            f[c] = f[c].map(lambda v: f"x{v:.2f}")
        elif c in ("median maxDD", "worst maxDD", "median XIRR after sell", "10th pct XIRR"):
            f[c] = f[c].map(lambda v: f"{v:.1%}")
        elif c in ("% behind 1x",):
            f[c] = f[c].map(lambda v: f"{v:.0f}%")
        elif c in ("median underwater y", "worst underwater y", "avg exposure"):
            f[c] = f[c].map(lambda v: f"{v:.1f}")
    return f.to_string(index=False)


# ------------------------------------------------------------------ PART B: real IUSA, daily
def real_inputs():
    px, dist = load_post_holdout()
    f = ROOT / "data" / "raw" / "macro" / "ecb_dfr.csv"
    if not f.exists():
        sys.exit("missing data/raw/macro/ecb_dfr.csv: run  python scripts/0_fetch_data.py --only ecb_dfr")
    dfr = pd.read_csv(f, parse_dates=["date"]).set_index("date")["dfr_pct"].sort_index() / 100
    c = px["close"]
    d_amt = (dist.set_index(pd.to_datetime(dist["ex_date"]))["amount_eur"].groupby(level=0).sum()
             .reindex(c.index).fillna(0.0))
    r1 = ((c + d_amt) / c.shift(1) - 1).fillna(0.0)
    rate = dfr.reindex(c.index.union(dfr.index)).ffill().reindex(c.index).fillna(0.0).shift(1).bfill()
    r2 = 2 * r1 - (rate + FUND_SPREAD) / 252 - FUND_TER / 252
    mrate = (rate + LOAN_SPREAD).clip(lower=0.0) / 252
    return c.index, r1, r2, mrate


def real_runs():
    idx, r1, r2, mr = real_inputs()
    rows, paths = [], {}
    for sk, s in REAL_STARTS.items():
        d = idx[idx >= pd.Timestamp(s)]
        q = pd.Series(d, index=d).groupby([d.year, d.quarter]).first().values
        cday = np.isin(d, q)
        k = np.cumsum(cday) - 1
        C = np.where(cday, plan_matt(k), 0.0)[None, :]
        R1, R2, MR = (x.loc[d].to_numpy()[None, :].copy() for x in (r1, r2, mr))
        R1[:, 0] = R2[:, 0] = MR[:, 0] = 0.0
        base = None
        for name, mode, lev in STRATS:
            res = simulate(R1, R2, MR, C, lev(k)[None, :], mode)
            mdd, under = drawdown_stats(res["nw_pre"], res["nw_post"], 252)
            fa = res["final_after"][0]
            base = fa if base is None else base
            fl = [(d[t], -C[0, t]) for t in range(len(d)) if C[0, t] > 0]
            ex = res["exposure"][0, 1:]
            rows.append({"start": sk, "strategy": name, "contributed": C.sum(), "final_before_sell": res["final_before"][0],
                         "final_after_sell": fa, "x_vs_1x": fa / base, "margin_call": not res["alive"][0],
                         "xirr_after_sell": xirr(fl + [(d[-1], fa)]) if res["alive"][0] else -1.0,
                         "max_drawdown": mdd[0], "longest_underwater_years": under[0],
                         "interest": res["interest"][0], "tax_on_rebalancing": res["tax_on_rebalancing"][0],
                         "exposure_min": np.nanmin(ex), "exposure_max": np.nanmax(ex)})
            paths[(sk, name)] = pd.Series(res["nw_post"][0], index=d)
    return pd.DataFrame(rows), paths


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    line = "=" * 100
    k = np.arange(YEARS * 4)
    c = plan_matt(k)
    print(f"\n{line}\nSTUDY 13 Matt's plan (POST-HOLDOUT, personal planning). 30 years, contributions per quarter:")
    print(f"  years 0-1: EUR 500 | years 1-4: EUR 1,200 | year 4: EUR 3,000, +5% a year -> year 29: EUR {c[-1]:,.0f}")
    print(f"  total paid in EUR {c.sum():,.0f}; first 5 years = EUR {c[:20].sum():,.0f} ({c[:20].sum() / c.sum():.1%} of it)")

    a, starts, ratios = shiller_windows(plan_matt)
    a.to_csv(OUT / "shiller_windows.csv", index=False)
    sa = summary(a)
    print(f"\n{line}\nPART A  Shiller S&P 500, {len(starts)} thirty-year windows, starts {starts[0]:%Y-%m} -> {starts[-1]:%Y-%m}"
          f"\n        (USD nominal, monthly; x vs 1x and XIRR are after selling everything, 26% tax)\n{line}")
    print(fmt(sa))
    base = a[a["strategy"] == STRATS[0][0]]["final_after_sell"]
    print(f"\n  1x final (nominal, after sell): median {base.median():,.0f} | 10th pct {base.quantile(.1):,.0f} | "
          f"worst {base.min():,.0f} | paid in {c.sum():,.0f}")
    for name in ("loan 1.25x", "ETF 75/25 (1.25x)", "loan 1.5x", "ETF 50/50 (1.5x)"):
        t = a[a["strategy"] == name].nsmallest(5, "x_vs_1x")
        print(f"  worst 5 starts, {name:<18}: " + ", ".join(f"{s:%Y-%m} x{x:.2f}" for s, x in zip(t["start"], t["x_vs_1x"])))
    by_dec = a.assign(decade=a["start"].dt.year // 10 * 10).pivot_table(index="decade", columns="strategy",
                                                                           values="x_vs_1x", aggfunc="median")
    print("\n  median x vs 1x by start decade:")
    print(by_dec[[s[0] for s in STRATS[1:]]].round(2).to_string())

    s_hi = summary(shiller_windows(plan_matt, extra_rate=0.02)[0])
    s_hi.to_csv(OUT / "shiller_summary_rates_plus2.csv", index=False)
    print(f"\n{line}\nPART A2  same, every borrowing rate +2 points (loan = 10y + 3.5%, inside the 2x fund = 10y + 2.5%)\n{line}")
    print(fmt(s_hi))
    s_flat = summary(shiller_windows(plan_flat, with_xirr=False)[0])
    s_flat.to_csv(OUT / "shiller_summary_flat500.csv", index=False)
    print(f"\n{line}\nPART A3  same, flat EUR 500 a quarter (the study's usual investor), for comparison\n{line}")
    print(fmt(s_flat))
    sa.to_csv(OUT / "shiller_summary.csv", index=False)

    b, paths = real_runs()
    b.to_csv(OUT / "real_iusa.csv", index=False)
    print(f"\n{line}\nPART B  real IUSA (EUR, daily), ECB deposit rate + 1.5% loan / + 0.5% inside the 2x fund, same plan\n{line}")
    for sk in REAL_STARTS:
        t = b[b["start"] == sk]
        print(f"\n  start {sk}: paid in EUR {t['contributed'].iloc[0]:,.0f}")
        for _, r in t.iterrows():
            if r["margin_call"]:
                print(f"    {r['strategy']:<20} MARGIN CALL -> 0")
                continue
            print(f"    {r['strategy']:<20} EUR {r['final_after_sell']:>10,.0f} after sell | x{r['x_vs_1x']:.2f} | "
                  f"XIRR {r['xirr_after_sell']:6.2%} | max DD {r['max_drawdown']:6.1%} | underwater {r['longest_underwater_years']:.1f}y"
                  f" | interest EUR {r['interest']:,.0f} | tax on rebalancing EUR {r['tax_on_rebalancing']:,.0f}"
                  f" | exposure {r['exposure_min']:.2f}-{r['exposure_max']:.2f}")
    charts(a, starts, ratios, b, paths)
    print(f"\nWROTE {OUT.relative_to(ROOT)}/*.csv | figures/fig28_my_plan_1871.png, fig29_my_plan_real.png")
    print("DONE. Paste everything back.")


def charts(a, starts, ratios, b, paths):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    plots.style()
    cols = {"1x (plain B&H)": plots.INK, "loan 1.25x": plots.SERIES[0], "loan 1.5x": plots.SERIES[1],
            "loan 1.5x -> 1.25x": plots.SERIES[6], "ETF 75/25 (1.25x)": plots.SERIES[0],
            "ETF 50/50 (1.5x)": plots.SERIES[1], "ETF 50/50 -> 75/25": plots.SERIES[6]}
    ls = {n: ((0, (4, 2)) if n.startswith("ETF") else "-") for n in cols}
    ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

    def ratio_axis(ax, lo, hi):
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"x{v:g}"))

    # fig28: x vs 1x by start date
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    yrs = np.array([d.year + (d.month - 1) / 12 for d in starts])
    alive = np.concatenate([r[r > 0] for n, r in ratios.items()])
    lo, hi = min(0.45, alive.min() * 0.95), max(1.1, alive.max() * 1.05)
    for ax, group, title in ((axes[0], ("loan 1.25x", "loan 1.5x", "loan 1.5x -> 1.25x"), "loan on top of one fund"),
                             (axes[1], ("ETF 75/25 (1.25x)", "ETF 50/50 (1.5x)", "ETF 50/50 -> 75/25"), "1x fund + 2x fund, no loan")):
        ax.axhline(1.0, color=plots.INK, linewidth=1.0)
        for name in group:
            r = ratios[name]
            ok = r > 0
            med = np.median(r)
            ax.plot(yrs[ok], r[ok], color=cols[name], linewidth=1.3, label=f"{name}  (median x{med:.2f})")
            if (~ok).any():
                ax.plot(yrs[~ok], np.full((~ok).sum(), lo * 1.05), "X", color=cols[name], markersize=6)
        ratio_axis(ax, lo, hi)
        ax.legend(loc="upper right", ncol=3)
        ax.set_ylabel("final money vs plain 1x (after tax)")
        ax.set_title(f"30 years of Matt's plan, by start date: {title}")
        ax.set_xlim(yrs[0], yrs[-1] + 1)
    axes[1].set_xlabel("start of the 30-year window")
    fig.tight_layout()
    plots.note(fig, "Shiller monthly S&P 500 total return (USD, monthly averages smooth crashes). Loan: 10y yield + 1.5%, re-set "
                    "every quarter, margin call = 0 (X at the bottom).\n2x fund: monthly 2x minus (10y + 0.5%) minus 1% yearly fund "
                    "costs, never sold. 26% tax on every sale; above the black line = leverage won.")
    plots.save(fig, FIG / "fig28_my_plan_1871.png")

    # fig29: real IUSA, each strategy's net worth divided by the 1x net worth, day by day
    show = ["loan 1.25x", "loan 1.5x", "loan 1.5x -> 1.25x", "ETF 75/25 (1.25x)", "ETF 50/50 (1.5x)", "ETF 50/50 -> 75/25"]
    fig, axes = plt.subplots(len(REAL_STARTS), 1, figsize=(12, 3.8 * len(REAL_STARTS)))
    for ax, sk in zip(axes, REAL_STARTS):
        base = paths[(sk, "1x (plain B&H)")]
        f1 = b[(b["start"] == sk) & (b["strategy"] == "1x (plain B&H)")]["final_after_sell"].iloc[0]
        ax.axhline(1.0, color=plots.INK, linewidth=1.0, label=f"1x (plain B&H)  EUR {f1:,.0f}")
        vals = []
        for name in show:
            s = (paths[(sk, name)] / base).where(base > 0)
            r = b[(b["start"] == sk) & (b["strategy"] == name)].iloc[0]
            lab = (f"{name}  MARGIN CALL" if r["margin_call"]
                   else f"{name}  EUR {r['final_after_sell']:,.0f} (x{r['x_vs_1x']:.2f})")
            ax.plot(s.index, s.values, color=cols[name], linestyle=ls[name], linewidth=1.3, label=lab)
            vals.append(s[s > 0].dropna())
        v = pd.concat(vals)
        lo, hi = max(0.3, v.min() * 0.95), max(1.1, v.max() * 1.05)
        ratio_axis(ax, lo, hi)
        first, lastd = base.index[[0, -1]]
        ax.set_xlim(first, lastd)
        ax.xaxis.set_major_locator(mdates.YearLocator(2 if (lastd - first).days > 3000 else 1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(loc="upper left", ncol=2, fontsize=7.5)
        ax.set_ylabel("net worth vs plain 1x")
        ax.set_title(f"start {sk}: Matt's plan in real IUSA (EUR = after selling everything at the end)")
    fig.tight_layout()
    plots.note(fig, "Solid = loan, dashed = 1x fund + 2x fund (same colour = same target). ECB deposit rate + 1.5% (loan) / + 0.5% "
                    "(2x fund) + 1% yearly fund costs.\nContributions: 500/q year 1, 1,200/q years 2-4, then 3,000/q +5% a year. "
                    "Early on the ratio swings a lot because the amounts are small.")
    plots.save(fig, FIG / "fig29_my_plan_real.png")


if __name__ == "__main__":
    main()
