"""
12_overlays.py  -  STUDY 10 (pre-registered §11): do TREND or VALUE add anything?
                   POST-HOLDOUT, exploratory (design + predictions in notes/decisions.md first).

Each overlay (src/overlays.py) is reported as the EXTRA return over the same setup without it:
  bases     quarterly B&H (added: it's the main benchmark), monthly DCA (pre-registered),
            E2 reference dip rule (pre-registered "reference strategy")
  overlays  10a trend: only buy above the 200-day EMA | only buy below it
            10b value: orders x1.5 cheap / x1 / x0.5 expensive (Matt's formula), unspent cash waits
  periods   1999-2026 (main) + separate starts 1999-2008, 2009-2018, 2019-2026
PLUS a direct check of the signals themselves: after each day the signal said cheap / neutral /
expensive (or up / down trend), what did IUSA actually return over the next 1 and 3 years?
That answers "can these tell me if we're in a good or bad regime?" without any trading rule.
Outputs: results/12_overlays/*.csv, figures/fig23_overlays.png, fig24_signals.png, fig25_forward.png
Run from project root:  python scripts/12_overlays.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import plots                                                       # noqa: E402
from src.benchmarks import BuyHold, MonthlyDCA                             # noqa: E402
from src.costs import IBKR_TIERED                                           # noqa: E402
from src.data import PROC, load_post_holdout                   # noqa: E402
from src.engine import Engine                                               # noqa: E402
from src.metrics import summarize                                           # noqa: E402
from src.overlays import Overlay, value_signal                             # noqa: E402
from src.pit import known_at_open_features                                  # noqa: E402
from src.signals import Rule                                                # noqa: E402
from src.strategy import DipStrategy                                        # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "12_overlays"
FIG = ROOT / "figures"
COST = IBKR_TIERED
PERIODS = {"1999-2026": ("1999-01-04", None), "1999-2008": ("1999-01-04", "2008-12-31"),
           "2009-2018": ("2009-01-01", "2018-12-31"), "2019-2026": ("2019-01-01", None)}
OVERLAYS = ["none", "trend_above", "trend_below", "value"]
NICE = {"trend_above": "only buy above 200-day EMA", "trend_below": "only buy below 200-day EMA",
        "value": "value sizing x1.5 / x1 / x0.5"}
pd.set_option("display.width", 220)


def yahoo_close(path):
    d = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    d.columns = [c.lower().replace(" ", "_") for c in d.columns]
    return d[~d.index.duplicated(keep="last")]["close"]


def e2_reference():
    f = pd.read_csv(ROOT / "results" / "07_plateau" / "finalists.csv")
    r = f[f["style"] == "E2"].iloc[0]
    re_ = r["reentry"]
    rule = (Rule(r["ref"], r["kind"], float(r["thr"]), "C", cooldown=int(re_[1:])) if str(re_).startswith("C")
            else Rule(r["ref"], r["kind"], float(r["thr"]), re_))
    dl = int(r["deadline"]) if str(r["deadline"]).isdigit() else str(r["deadline"])
    return (lambda: DipStrategy(rule, "E2", int(r["pieces"]), dl, 0.0)), f"E2 ref {r['rule']} P{int(r['pieces'])} D{r['deadline']}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px, dist = load_post_holdout()
    feats = known_at_open_features(px)
    above = feats["prev_close"] > feats["ema_200"]
    below = feats["prev_close"] < feats["ema_200"]
    tnx_f = RAW / "macro" / "TNX_yahoo.csv"
    val = value_signal(yahoo_close(RAW / "etf" / "GSPC.csv"), pd.read_parquet(PROC / "shiller_monthly.parquet"),
                       pd.read_parquet(PROC / "pit_shiller.parquet"),
                       yahoo_close(tnx_f) if tnx_f.exists() else None)
    val.to_csv(OUT / "value_signal_daily.csv")
    sm = pd.read_parquet(PROC / "shiller_monthly.parquet")["P"]
    spx_m = yahoo_close(RAW / "etf" / "GSPC.csv").resample("MS").mean()
    scale = (spx_m / sm).dropna().loc["1999":]

    make_ref, ref_name = e2_reference()
    bases = [("quarterly B&H", lambda: BuyHold()), ("monthly DCA", lambda: MonthlyDCA(1)), (ref_name, make_ref)]
    rows = []
    for p, (a, b) in PERIODS.items():
        start, end = pd.Timestamp(a), (pd.Timestamp(b) if b else px.index[-1])
        for bname, make in bases:
            for ov in OVERLAYS:
                inner = make()
                pol = inner if ov == "none" else Overlay(
                    inner, ov, trend_ok=above if ov == "trend_above" else below if ov == "trend_below" else None,
                    mult=val["mult"] if ov == "value" else None)
                eng = Engine(px, dist, start, end, pol, cost=COST)
                res = eng.run()
                assert abs(res.totals["books_residual"]) < 1e-6, pol.name
                m = summarize(res, COST)
                m.update({"period": p, "base": bname, "overlay": ov})
                rows.append(m)
    df = pd.DataFrame(rows)
    for col in ("xirr_before_sell", "xirr_after_sell"):
        ref = df[df["overlay"] == "none"].set_index(["period", "base"])[col]
        df["extra_" + col] = df[col] - df.set_index(["period", "base"]).index.map(ref).to_numpy()
    df["study"] = "10_overlays"
    df.to_csv(OUT / "runs.csv", index=False)

    # ---------------- do the signals predict what comes next?
    s, e = pd.Timestamp("1999-01-04"), px.index[-1]
    c = px["close"]
    d_amt = dist.set_index(pd.to_datetime(dist["ex_date"]))["amount_eur"].groupby(level=0).sum().reindex(c.index).fillna(0)
    tr = ((c + d_amt) / c.shift(1)).fillna(1).cumprod()                        # IUSA total-return index, EUR
    fwd = {}
    for yrs in (1, 3):
        h = 252 * yrs
        f = (tr.shift(-h) / tr) ** (1 / yrs) - 1
        fwd[yrs] = f
    state_v = val["mult"].map({1.5: "cheap", 1.0: "neutral", 0.5: "expensive"}).reindex(c.index)
    state_t = pd.Series(np.where(above, "uptrend", np.where(below, "downtrend", None)), index=c.index)
    fw_rows = []
    for sig, st in (("value", state_v), ("trend", state_t)):
        for yrs, f in fwd.items():
            d = pd.DataFrame({"state": st, "fwd": f}).loc[s:e].dropna()
            for k, g in d.groupby("state"):
                fw_rows.append({"signal": sig, "years": yrs, "state": k, "days": len(g), "share_of_days": len(g) / len(d),
                                "median_fwd": g["fwd"].median(), "mean_fwd": g["fwd"].mean(), "pct_negative": (g["fwd"] < 0).mean()})
            fw_rows.append({"signal": sig, "years": yrs, "state": "all days", "days": len(d), "share_of_days": 1.0,
                            "median_fwd": d["fwd"].median(), "mean_fwd": d["fwd"].mean(), "pct_negative": (d["fwd"] < 0).mean()})
    FW = pd.DataFrame(fw_rows)
    FW.to_csv(OUT / "forward_returns.csv", index=False)

    # ---------------- print
    line = "=" * 78
    print(f"\n{line}\nSTUDY 10 overlays (POST-HOLDOUT). EXTRA XIRR from adding the overlay, before selling (pts/yr)\n{line}")
    vv = val.loc[s:e]
    print(f"sanity: Yahoo S&P monthly average / Shiller P = {scale.median():.3f} (should be ~1.00) | "
          f"e median {vv['e'].median():.1%}, r median {vv['r'].median():.1%}, e - r range "
          f"{vv['e_minus_r'].min():+.1%} .. {vv['e_minus_r'].max():+.1%}")
    print(f"value signal 1999-{e.year}: cheap {np.mean(vv['mult'] == 1.5):.0%} of days | neutral {np.mean(vv['mult'] == 1.0):.0%} "
          f"| expensive {np.mean(vv['mult'] == 0.5):.0%}  ||  trend: above EMA {above.loc[s:e].mean():.0%} of days")
    print("\n" + f"{'base':38s}{'overlay':32s}" + "".join(f"{p:>11s}" for p in PERIODS))
    for bname, _ in bases:
        for ov in OVERLAYS[1:]:
            t = df[(df["base"] == bname) & (df["overlay"] == ov)].set_index("period")
            print(f"{bname[:37]:38s}{NICE[ov]:32s}" + "".join(f"{100 * t.loc[p, 'extra_xirr_before_sell']:>+11.2f}" for p in PERIODS))
    t = df[(df["period"] == "1999-2026") & (df["overlay"] != "none")]
    print(f"\n1999-2026, after selling: extra XIRR median {100 * t['extra_xirr_after_sell'].median():+.2f} | "
          f"best {100 * t['extra_xirr_after_sell'].max():+.2f} | worst {100 * t['extra_xirr_after_sell'].min():+.2f}")
    vb = df[(df["period"] == "1999-2026") & (df["overlay"] == "value")]
    print("value sizing, average cash held: " + " | ".join(f"{r['base'][:14]} {r['avg_cash_pct']:.1%}" for _, r in vb.iterrows()))
    print(f"\n{line}\nDO THE SIGNALS PREDICT? IUSA total return (EUR) over the NEXT 1 / 3 years, by signal state, 1999-{e.year}\n{line}")
    for sig in ("value", "trend"):
        print(f"\n{sig}:")
        for yrs in (1, 3):
            g = FW[(FW["signal"] == sig) & (FW["years"] == yrs)]
            order = ["cheap", "neutral", "expensive", "uptrend", "downtrend", "all days"]
            g = g.assign(o=g["state"].map(order.index)).sort_values("o")
            print(f"  next {yrs}y: " + " | ".join(f"{r['state']} {r['median_fwd']:+.1%} ({r['share_of_days']:.0%} of days, "
                                                 f"{r['pct_negative']:.0%} negative)" for _, r in g.iterrows()))
    last = val.dropna(subset=["e_minus_r"]).iloc[-1]
    print(f"\nlatest reading ({val.dropna(subset=['e_minus_r']).index[-1].date()}): expected return e {last['e']:.1%} vs required "
          f"r {last['r']:.1%} -> e - r {last['e_minus_r']:+.1%} -> "
          f"{ {1.5: 'CHEAP', 1.0: 'NEUTRAL', 0.5: 'EXPENSIVE'}[last['mult']] } | trend: "
          f"{'above' if above.iloc[-1] else 'below'} its 200-day EMA")
    print("  (a description of the signal today, not advice; see the forward-return table for how much it ever meant)")
    print("\nNote: forward windows overlap, so the day counts overstate how many independent observations there are.")
    charts(df, val, above, px, FW, s, e)
    print(f"\nWROTE {OUT.relative_to(ROOT)}/runs.csv, forward_returns.csv, value_signal_daily.csv")
    print("      figures/fig23_overlays.png, fig24_signals.png, fig25_forward.png\nDONE. Paste everything back.")


def charts(df, val, above, px, FW, s, e):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    plots.style()
    C = plots.SERIES
    better, worse = "#2a78d6", "#e34948"

    # fig23 extra XIRR per base x overlay, 4 periods
    combos = [(b, o) for b in df["base"].unique() for o in OVERLAYS[1:]]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
    for ax, p in zip(axes, PERIODS):
        v = [df[(df["period"] == p) & (df["base"] == b) & (df["overlay"] == o)]["extra_xirr_before_sell"].iloc[0] * 100
             for b, o in combos]
        y = np.arange(len(combos))[::-1]
        ax.barh(y, v, color=[better if x > 0 else worse for x in v], height=0.6)
        ax.axvline(0, color=plots.INK, linewidth=1)
        lim = np.nanmax(np.abs(v)) * 1.5 + 0.05
        ax.set_xlim(-lim, lim)
        for yi, x in zip(y, v):
            ax.text(x, yi, f" {x:+.2f}" if x >= 0 else f"{x:+.2f} ", va="center", ha="left" if x >= 0 else "right",
                    fontsize=7, color=plots.INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{b[:22]}: {NICE[o]}" for b, o in combos], fontsize=7.5)
        ax.grid(axis="y", visible=False)
        ax.set_title(p)
        ax.set_xlabel("extra XIRR vs same setup without it (pts/yr)", fontsize=8)
    fig.suptitle("Do trend or value add anything? Extra return from each overlay", fontsize=11, weight="bold", color=plots.INK)
    fig.tight_layout()
    plots.note(fig, "Before selling. Each period is a separate start (1999-2008 ends at the 2008 low, 2009-2018 starts inside it). "
                    "Scales differ per panel.")
    plots.save(fig, FIG / "fig23_overlays.png")

    # fig24 the two signals through time
    c = px["close"].loc[s:e]
    v = val.loc[s:e]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    a1.plot(c.index, c.values, color=C[0], linewidth=1.2)
    a1.set_yscale("log")
    dn = (~above.loc[s:e]).astype(int)
    a1.fill_between(c.index, c.min() * 0.95, c.max() * 1.05, where=dn.values == 1, color=plots.GRID, alpha=0.9, linewidth=0)
    a1.set_ylim(c.min() * 0.95, c.max() * 1.05)
    a1.set_ylabel("IUSA (EUR, log)")
    a1.set_title("Grey = below its 200-day EMA (downtrend). Bottom: Matt's value signal, expected minus required return")
    em = v["e_minus_r"] * 100
    a2.plot(em.index, em.values, color=C[1], linewidth=1.2)
    a2.axhline(1, color=better, linewidth=1)
    a2.axhline(-1, color=worse, linewidth=1)
    a2.axhline(0, color=plots.INK2, linewidth=0.6)
    a2.text(em.index[0], 1.2, " cheap above this line (x1.5)", color=better, fontsize=7.5, va="bottom")
    a2.text(em.index[0], -1.2, " expensive below this line (x0.5)", color=worse, fontsize=7.5, va="top")
    a2.set_ylabel("e - r (pts)")
    a1.yaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    a1.yaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    a2.set_xlim(c.index[0], c.index[-1])
    a2.xaxis.set_major_locator(mdates.YearLocator(3))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plots.note(fig, "Both signals use only information available on the day (prices up to yesterday, Shiller data with a 6-month lag).")
    plots.save(fig, FIG / "fig24_signals.png")

    # fig25 forward returns by state
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, sig in zip(axes, ("value", "trend")):
        g = FW[FW["signal"] == sig]
        states = [x for x in (["cheap", "neutral", "expensive"] if sig == "value" else ["uptrend", "downtrend"]) + ["all days"]
                  if x in set(g["state"])]
        xs = np.arange(len(states))
        for j, yrs in enumerate((1, 3)):
            vals = [g[(g["years"] == yrs) & (g["state"] == st)]["median_fwd"].iloc[0] * 100 for st in states]
            ax.bar(xs + (j - 0.5) * 0.38, vals, width=0.36, color=C[j], label=f"next {yrs} year{'s' if yrs > 1 else ''} (per year)")
            for x, v_ in zip(xs + (j - 0.5) * 0.38, vals):
                ax.text(x, v_, f"{v_:+.1f}%", ha="center", va="bottom" if v_ >= 0 else "top", fontsize=7.5, color=plots.INK2)
        ax.axhline(0, color=plots.INK, linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(states)
        ax.grid(axis="x", visible=False)
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(min(lo_, 0), hi_ + 0.3 * (hi_ - min(lo_, 0)))
        ax.legend(loc="upper left", ncols=2)
        ax.set_title("value signal" if sig == "value" else "200-day EMA trend")
        ax.set_ylabel("median IUSA return per year (EUR)")
    plots.note(fig, f"{s.year}-{e.year}, IUSA total return in EUR after each day in that state. Overlapping windows: "
                    "fewer independent observations than days.")
    plots.save(fig, FIG / "fig25_forward.png")


if __name__ == "__main__":
    main()
