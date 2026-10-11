"""
13_leverage.py  -  STUDY 09: leverage on EUR 500 a quarter (curiosity, POST-HOLDOUT, exploratory).

Three separate starts (2000, 2010, 2020 -> last day), every leverage 1x (= quarterly B&H),
1.2x, 1.5x, 2x, 3x, 4x, 5x, 10x, two ways (src/leverage.py):
  MARGIN LOAN      target leverage re-set every quarter, ECB rate + 1.5%, margin call = zero
  LEVERAGED ETF    daily reset (pre-registered formula), wiped out by one bad enough day
Outputs: results/13_leverage/*.csv, figures/fig26_leverage_margin.png, fig27_leverage_etf.png
Run from project root:  python scripts/13_leverage.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import plots                                           # noqa: E402
from src.benchmarks import BuyHold                              # noqa: E402
from src.costs import IBKR_TIERED, with_fractional                               # noqa: E402
from src.data import load_post_holdout                          # noqa: E402
from src.engine import Engine                                   # noqa: E402
from src.leverage import LeverageTarget, letf_prices, maintenance  # noqa: E402
from src.metrics import summarize                               # noqa: E402

OUT = ROOT / "results" / "13_leverage"
FIG = ROOT / "figures"
COST = IBKR_TIERED
STARTS = {"2000": "2000-01-01", "2010": "2010-01-01", "2020": "2020-01-01"}
LEVELS = (1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0)
NO_DIST = pd.DataFrame({"ex_date": pd.to_datetime([]), "pay_date": pd.to_datetime([]), "amount_eur": []})
pd.set_option("display.width", 220)


def load_rate():
    f = ROOT / "data" / "raw" / "macro" / "ecb_dfr.csv"
    if not f.exists():
        sys.exit("missing data/raw/macro/ecb_dfr.csv: run  python scripts/0_fetch_data.py --only ecb_dfr")
    return pd.read_csv(f, parse_dates=["date"]).set_index("date")["dfr_pct"].sort_index() / 100


def bars(s):
    return pd.DataFrame({"open": s, "high": s, "low": s, "close": s}, index=s.index)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px, dist = load_post_holdout()
    dfr = load_rate()
    margin_rate = (dfr + 0.015).clip(lower=0.0)
    end = px.index[-1]
    rows, paths = [], {}
    for sk, s in STARTS.items():
        start = pd.Timestamp(s)
        for L in LEVELS:
            # ---- margin loan
            pol = LeverageTarget(L)
            eng = Engine(px, dist, start, end, pol, cost=COST, debt_rate=margin_rate)
            res = eng.run()
            assert abs(res.totals["books_residual"]) < 1e-6, pol.name
            m = summarize(res, COST)
            net = res.daily["value"] - res.daily["debt"]
            busted = pol.bust is not None
            if busted:
                net = net.loc[:pol.bust]
                contributed = float(res.daily["contributed"].loc[:pol.bust].iloc[-1])
            else:
                contributed = m["contributed"]
            rows.append({"model": "margin loan", "start": sk, "L": L, "busted": pol.bust.date() if busted else None,
                         "final_net_worth": 0.0 if busted else m["final_value"],
                         "after_sell": 0.0 if busted else m["final_value"] - max(res.totals["value_end"] - res.totals["cash_end"] - res.totals["cost_basis"], 0) * COST.cgt,
                         "contributed": contributed, "xirr_before_sell": -1.0 if busted else m["xirr_before_sell"],
                         "xirr_after_sell": -1.0 if busted else m["xirr_after_sell"],
                         "max_drawdown": m["max_drawdown"] if not busted else -1.0,
                         "residual_at_call": pol.residual if busted else np.nan,
                         "max_leverage_seen": pol.max_lev, "interest_paid": res.totals["interest"],
                         "tax_on_rebalancing": res.totals["cgt_paid"], "margin_call_level": maintenance(L)})
            paths[("margin loan", sk, L)] = (net, pol.bust)
            # ---- leveraged ETF
            price, wipe = letf_prices(px, dist, L, dfr, start, end)
            stop = price.index[price.index.get_loc(wipe) - 1] if wipe is not None else price.index[-1]
            # fractional shares: a decayed fund can trade at fractions of a cent
            e2 = Engine(bars(price.loc[:stop]), NO_DIST, start, stop, BuyHold(), cost=with_fractional(COST))
            r2 = e2.run()
            m2 = summarize(r2, COST)
            v2 = r2.daily["value"]
            rows.append({"model": "leveraged ETF", "start": sk, "L": L, "busted": wipe.date() if wipe is not None else None,
                         "final_net_worth": 0.0 if wipe is not None else m2["final_value"],
                         "after_sell": 0.0 if wipe is not None else m2["final_value"] - max(r2.totals["value_end"] - r2.totals["cash_end"] - r2.totals["cost_basis"], 0) * COST.cgt,
                         "contributed": m2["contributed"], "xirr_before_sell": -1.0 if wipe is not None else m2["xirr_before_sell"],
                         "xirr_after_sell": -1.0 if wipe is not None else m2["xirr_after_sell"],
                         "max_drawdown": -1.0 if wipe is not None else m2["max_drawdown"],
                         "residual_at_call": np.nan, "max_leverage_seen": L, "interest_paid": np.nan,
                         "tax_on_rebalancing": 0.0, "margin_call_level": np.nan})
            paths[("leveraged ETF", sk, L)] = (v2, wipe)
    df = pd.DataFrame(rows)
    df["study"] = "09_leverage"
    df.to_csv(OUT / "runs.csv", index=False)

    line = "=" * 78
    print(f"\n{line}\nSTUDY 09 leverage (POST-HOLDOUT, curiosity). EUR 500/quarter, IUSA, IBKR Tiered\n{line}")
    for model in ("margin loan", "leveraged ETF"):
        print(f"\n{model.upper()}" + ("  (re-set to L every quarter, ECB rate + 1.5%, margin call = zero)" if model == "margin loan"
                                     else "  (daily reset, pre-registered formula, a bad enough day = zero)"))
        for sk in STARTS:
            t = df[(df["model"] == model) & (df["start"] == sk)]
            base = t[t["L"] == 1.0].iloc[0]
            print(f"\n  start {sk}  (1x = quarterly B&H in this model: EUR {base['final_net_worth']:,.0f} from EUR {base['contributed']:,.0f}, "
                  f"XIRR {base['xirr_before_sell']:.2%})")
            for _, r in t.iterrows():
                if r["L"] == 1.0:
                    continue
                if r["busted"] is not None:
                    extra = (f" | forced sale left EUR {r['residual_at_call']:,.0f}, counted as 0"
                             if model == "margin loan" else "")
                    print(f"    {r['L']:>4g}x  BUSTED {r['busted']}  after putting in EUR {r['contributed']:,.0f}{extra}")
                else:
                    print(f"    {r['L']:>4g}x  EUR {r['final_net_worth']:>12,.0f} | x{r['final_net_worth'] / base['final_net_worth']:.2f} vs 1x"
                          f" | XIRR {r['xirr_before_sell']:6.2%} (after sell {r['xirr_after_sell']:6.2%}) | max drawdown "
                          f"{r['max_drawdown']:6.1%}" + (f" | interest EUR {r['interest_paid']:,.0f} | peak leverage "
                                                         f"{r['max_leverage_seen']:.1f}x" if model == "margin loan" else ""))
    charts(df, paths)
    print(f"\nWROTE {OUT.relative_to(ROOT)}/runs.csv | figures/fig26_leverage_margin.png, fig27_leverage_etf.png")
    print("DONE. Paste everything back.")


def charts(df, paths):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    plots.style()
    cols = {1.0: plots.INK, 1.2: plots.SERIES[0], 1.5: plots.SERIES[2], 2.0: plots.SERIES[3], 3.0: plots.SERIES[1],
            4.0: plots.SERIES[4], 5.0: plots.SERIES[6], 10.0: plots.SERIES[7]}
    for model, fname in (("margin loan", "fig26_leverage_margin.png"), ("leveraged ETF", "fig27_leverage_etf.png")):
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        for ax, sk in zip(axes, STARTS):
            ends = []
            for L in LEVELS:
                s, bust = paths[(model, sk, L)]
                s = s[s > 0]
                if not len(s):
                    continue
                ax.plot(s.index, s.values, color=cols[L], linewidth=2.0 if L == 1 else 1.3)
                if bust is not None:
                    ax.plot([s.index[-1]], [s.values[-1]], marker="X", color=cols[L], markersize=9)
                    lab = f"{L:g}x busted {pd.Timestamp(bust):%b %Y}"
                else:
                    r = df[(df["model"] == model) & (df["start"] == sk) & (df["L"] == L)].iloc[0]
                    lab = f"{L:g}x  EUR {r['final_net_worth']:,.0f}"
                ends.append((s.values[-1], lab, cols[L], s.index[-1], bust is not None))
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ylo, yhi = (np.log10(v) for v in ax.get_ylim())
            gap = 0.045 * (yhi - ylo)
            alive = sorted([e for e in ends if not e[4]], key=lambda e: e[0])
            ys, last = [], -np.inf
            for v, *_ in alive:
                y = max(np.log10(v), last + gap)
                ys.append(y)
                last = y
            for (v, lab, c, x, _), y in zip(alive, ys):
                ax.text(x, 10 ** y, f"  {lab}", color=c, fontsize=7.5, va="center")
            for k, (v, lab, c, x, b) in enumerate([e for e in ends if e[4]]):
                ax.annotate(lab, (x, v), textcoords="offset points", xytext=(6, -10 - 11 * (k % 3)), fontsize=7, color=c)
            first = min(paths[(model, sk, 1.0)][0].index)
            last_d = max(paths[(model, sk, 1.0)][0].index)
            ax.set_xlim(first, last_d + (last_d - first) * 0.22)
            ax.xaxis.set_major_locator(mdates.YearLocator(2 if sk != "2020" else 1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.set_xticks([t for t in ax.get_xticks() if t <= mdates.date2num(last_d)])
            ax.spines["bottom"].set_bounds(mdates.date2num(first), mdates.date2num(last_d))
            ax.set_ylabel("net worth (EUR, log)")
            ax.set_title(f"start {sk}: EUR 500 a quarter, {model}")
        fig.tight_layout()
        plots.note(fig, ("Re-set to the target leverage every quarter, ECB rate + 1.5%. X = margin call (equity below min(25%, half the "
                         "starting margin)), counted as zero." if model == "margin loan" else
                         "Daily reset: L x daily return - (L-1) x (ECB rate + 0.5%) - 0.6% fee per year. X = a day that wiped the fund out."))
        plots.save(fig, FIG / fname)


if __name__ == "__main__":
    main()
