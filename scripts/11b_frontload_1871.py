"""
11b_frontload_1871.py  -  STUDY 11b: out-of-sample check of the one result that looked good in fig19.

In study 11 (1999-2026, IUSA in EUR) "front-load at -15%" was the only trigger with a blue row,
but it beat at most 70-88% of random-date front-loads, and its neighbours (-10%, -20%) didn't
agree. The -15% row was picked AFTER seeing the data, so the only fair test is data it never saw:
Robert Shiller's monthly S&P 500, 1871-1998 (128 years, before our sample starts).

Rules (fixed in notes/decisions.md before running, identical to study 11 where possible):
  - USD 500 every quarter (Jan, Apr, Jul, Oct), invested at that month's price.
  - Drawdown = this month's price vs the highest price so far. Front-load at -15%: invest the
    next N quarters (N = 1, 2, 4, 8) NOW, borrowed from the emergency fund (0% interest),
    repaid by skipping the next contributions. One-shot per crash, re-armed at a new high.
    At most 8 quarters outstanding. Decided on month t, invested at month t+1 (no look-ahead).
  - Placebo: the same number of front-loads on random months, 200x per size.
  - SUCCESS = the crash-timed version beats >= 90% of random dates in at least 3 of the 4 sizes.
Simplifications (same for every line, so comparisons stay fair): Shiller P is the monthly
AVERAGE price (smooths crashes a little), dividends reinvested monthly (D/12), 0.1% cost per buy,
no taxes, USD.
Also shown (descriptive): 30-year windows starting every 10 years, and the per-crash table.
Outputs: results/11b_frontload_1871/*.csv, figures/fig22_frontload_1871.png
Run from project root:  python scripts/11b_frontload_1871.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import plots                      # noqa: E402
from src.data import PROC                  # noqa: E402
from src.metrics import xirr               # noqa: E402

OUT = ROOT / "results" / "11b_frontload_1871"
FIG = ROOT / "figures"
X, SIZES, CAP, Q, FEE = 0.15, (1, 2, 4, 8), 8, 500.0, 0.001
PLACEBO, SEED = 200, 2026
SPAN = ("1871-01-01", "1998-12-01")
WINDOWS = [(y, y + 29) for y in range(1871, 1962, 10)]


def simulate(P, D, ath, dates, N, fire_at=None):
    """One investor, month by month. fire_at = set of month indices (placebo); None = -15% rule."""
    shares = debt = 0.0
    flows, fires = [], []
    armed, pending = True, False
    for t, d in enumerate(dates):
        p = P[t]
        if shares and D[t] == D[t]:                                  # dividends reinvested (skip NaN)
            shares += shares * D[t] / 12 * (1 - FEE) / p
        inv = 0.0
        if d.month in (1, 4, 7, 10):
            flows.append((d, -Q))
            rep = min(Q, debt)
            debt -= rep
            inv = Q - rep
        fire = (t in fire_at) if fire_at is not None else pending
        pending = False
        if fire and N > 0:
            amt = min(N * Q, CAP * Q - debt)
            if amt >= 0.5 * Q:
                debt += amt
                inv += amt
                fires.append((d, p / ath[t] - 1, amt, t))
        if inv > 0:
            shares += inv * (1 - FEE) / p
        if fire_at is None:
            if p >= ath[t] - 1e-12:
                armed = True
            elif armed and p / ath[t] - 1 <= -X and t + 1 < len(dates):
                armed, pending = False, True
    terminal = shares * P[-1] - debt
    return xirr(flows + [(dates[-1], terminal)]), fires


def run_span(sh, a, b, rng, sizes=SIZES):
    s = sh.loc[a:b]
    P, D = s["P"].to_numpy(float), s["D"].to_numpy(float)
    ath = sh["P"].cummax().loc[a:b].to_numpy(float)                   # history before the start counts
    dates = s.index
    base, _ = simulate(P, D, ath, dates, 0)
    out = []
    for N in sizes:
        real, fires = simulate(P, D, ath, dates, N)
        k = len(fires)
        plac = []
        for _ in range(PLACEBO if k else 0):
            days = set(rng.choice(len(dates) - 1, size=k, replace=False).tolist())
            plac.append(simulate(P, D, ath, dates, N, days)[0] - base)
        plac = np.array(plac)
        out.append({"start": a[:4], "end": b[:4], "N": N, "fires": k, "base_xirr": base,
                    "d_xirr": real - base, "placebo_median": np.median(plac) if k else np.nan,
                    "placebo_pct": np.mean(plac < real - base) if k else np.nan, "_plac": plac, "_fires": fires})
    return out, (P, dates)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sh = pd.read_parquet(PROC / "shiller_monthly.parquet")[["P", "D"]].dropna(subset=["P"])
    rng = np.random.default_rng(SEED)

    full, (P, dates) = run_span(sh, *SPAN, rng)
    wins = []
    for a, b in WINDOWS:
        w, _ = run_span(sh, f"{a}-01-01", f"{b}-12-01", rng)
        wins += w
    check, _ = run_span(sh, "1999-01-01", str(sh.index[-1].date()), rng, sizes=(4,))

    strip = lambda rows: pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])  # noqa: E731
    F, W = strip(full), strip(wins)
    F.to_csv(OUT / "full_span.csv", index=False)
    W.to_csv(OUT / "windows.csv", index=False)

    # per-crash table (N = 4, like study 11's representative config)
    rep = next(r for r in full if r["N"] == 4)
    rows = []
    for d0, dd0, amt, t in rep["_fires"]:
        later = [i for i in range(t + 1, len(dates)) if dates[i].month in (1, 4, 7, 10)][:int(np.ceil(amt / Q))]
        if later:
            pl = P[later].mean()
            rows.append({"date": f"{d0:%Y-%m}", "drawdown": dd0, "usd": amt, "price_vs_scheduled": P[t] / pl - 1,
                         "usd_gain": amt * (pl / P[t] - 1)})
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "per_crash.csv", index=False)

    line = "=" * 78
    print(f"\n{line}\nSTUDY 11b  front-load at -15%, Shiller S&P 500 {SPAN[0][:4]}-{SPAN[1][:4]} (data the rule never saw)\n{line}")
    print(f"quarterly B&H XIRR {F['base_xirr'].iloc[0]:.2%} (USD, dividends reinvested, no tax)\n")
    print("size        front-loads   vs B&H (pts/yr)   random dates: median   beats random")
    for _, r in F.iterrows():
        print(f"  N={r['N']}          {r['fires']:3d}          {100 * r['d_xirr']:+.3f}            "
              f"{100 * r['placebo_median']:+.3f}             p{100 * r['placebo_pct']:.0f}")
    passed = int((F["placebo_pct"] >= 0.90).sum())
    print(f"\nPRE-REGISTERED VERDICT: {passed} of 4 sizes beat >= 90% of random dates -> "
          + ("PASS: the -15% signal holds up out of sample" if passed >= 3 else
             "FAIL: the -15% row in study 11 was luck"))
    print("\n30-year windows (descriptive), beats-random percentile:")
    piv = W.pivot(index="N", columns="start", values="placebo_pct")
    print("  N  " + "".join(f"{c:>7s}" for c in piv.columns))
    for N, r in piv.iterrows():
        print(f"  {N}  " + "".join(f"{'' if np.isnan(v) else f'p{100 * v:.0f}':>7s}" for v in r.values))
    print(f"  windows with p >= 90 (all sizes): {(W['placebo_pct'] >= 0.9).sum()} of {W['placebo_pct'].notna().sum()} "
          f"| median p{100 * W['placebo_pct'].median():.0f}")
    print("\nper crash (N = 4; drawdown shown on the month the money went in): price paid vs the scheduled buys it replaced")
    for _, r in C.iterrows():
        print(f"  {r['date']}  drawdown {r['drawdown']:6.1%}  USD {r['usd']:,.0f}  paid {r['price_vs_scheduled']:+6.1%}"
              f"  -> {r['usd_gain']:+,.0f}")
    print(f"  helped {int((C['usd_gain'] > 0).sum())} times, hurt {int((C['usd_gain'] < 0).sum())} times | "
          f"total {C['usd_gain'].sum():+,.0f} USD")
    c = check[0]
    print(f"\nconsistency check, same monthly model on 1999-{c['end']} (USD, already-seen data): N=4 "
          f"{100 * c['d_xirr']:+.3f} pts/yr, p{100 * c['placebo_pct']:.0f}  (study 11, daily EUR: -0.00, p31)")
    charts(full, W)
    print(f"\nWROTE {OUT.relative_to(ROOT)}/full_span.csv, windows.csv, per_crash.csv | figures/fig22_frontload_1871.png")
    print("DONE. Paste everything back.")


def charts(full, W):
    import matplotlib.pyplot as plt
    plots.style()
    C = plots.SERIES
    fig = plt.figure(figsize=(14, 6.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1], hspace=0.55)
    for j, r in enumerate(full):
        ax = fig.add_subplot(gs[0, j])
        v = r["_plac"] * 100
        if len(v):
            ax.hist(v, bins=25, color=plots.MUTED, edgecolor=plots.SURFACE, linewidth=0.4)
        ax.axvline(0, color=plots.INK, linewidth=0.8)
        ax.axvline(r["d_xirr"] * 100, color=C[0], linewidth=2)
        ax.set_title(f"N = {r['N']} quarters: p{100 * r['placebo_pct']:.0f}", fontsize=9)
        ax.set_xlabel("vs B&H (pts/yr)", fontsize=8)
        if j == 0:
            ax.set_ylabel("random-date runs")
    ax = fig.add_subplot(gs[1, :])
    piv = W.pivot(index="N", columns="start", values="placebo_pct") * 100
    im = ax.imshow(piv.values, cmap=plots.DIVERGING, vmin=0, vmax=100, aspect="auto")
    for a in range(piv.shape[0]):
        for b in range(piv.shape[1]):
            v = piv.values[a, b]
            ax.text(b, a, "" if np.isnan(v) else f"p{v:.0f}", ha="center", va="center", fontsize=8, color=plots.INK)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"{s}-{int(s) + 29}" for s in piv.columns], fontsize=8)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([f"N = {n}" for n in piv.index])
    ax.grid(False)
    ax.set_title("30-year windows: did the -15% front-load beat random dates? (p50 = no better than random, blue = better)",
                 fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01).set_label("beats % of random dates")
    fig.suptitle("Out of sample, 1871-1998: front-loading at -15% vs the same front-loads on random months",
                 fontsize=11, weight="bold", color=plots.INK)
    plots.note(fig, "Top: grey = 200 random-date runs, blue line = crash-timed, 1871-1998. Shiller monthly S&P 500, USD, "
                    "dividends reinvested, 0.1% cost, no tax.")
    plots.save(fig, FIG / "fig22_frontload_1871.png")


if __name__ == "__main__":
    main()
