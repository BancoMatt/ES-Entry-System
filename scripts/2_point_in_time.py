"""
2_point_in_time.py  -  builds the ONE clean daily price history the engine will use,
plus the dividend and slow-data tables, all with timing rules applied.

READS raw + processed inputs. WRITES data/processed/pit_*.parquet only.

THE PRICE HISTORY IS STITCHED FROM THREE ERAS (column `era`):
  1. synthetic  1999-01-04 -> 2002-03-14  fund didn't exist yet: S&P 500 (price index)
                                          converted to EUR, scaled so it lands exactly on
                                          the fund's launch NAV (11.66 USD on 2002-03-15).
  2. nav        2002-03-15 -> 2008-12-31  the REAL fund's official daily NAV (iShares file),
                                          USD -> EUR with the ECB rate.
  3. real       2009-01-02 -> today       REAL IUSA prices on Euronext Amsterdam (Yahoo),
                                          cleaned. Yahoo's 2008 IUSA rows are frozen junk
                                          (zero volume, open=high=low=close) and are dropped.
  In eras 1-2 only a daily close is real; open/high/low are PROXIED from the S&P's own
  daily open/high/low ratios (US trading hours, not Amsterdam hours). Labelled, and
  validated on 2009+ where both exist.

DIVIDENDS (pit_distributions.parquet)
  real (2002+): iShares list, amounts in USD -> EUR at the ECB rate on the ex-date.
                Cash is credited on the PAY date (~2 weeks after ex-date), never earlier.
  synthetic (1999-2002): implied from S&P total-return vs price index, x0.85
                (15% US withholding tax inside an Irish fund). Paid quarterly. Labelled.

SLOW DATA
  Shiller: each month usable only from month + 1 + 6 months (lag fixed up front).
  10y yield, FX: used from the NEXT day (shift 1) in features.

Ends with an automatic LOOK-AHEAD TEST on the real data. Must print PASS.

Run from project root:  python scripts/2_point_in_time.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pit import (known_at_open_features, shiller_available, lookahead_check,  # noqa: E402
                     SHILLER_LAG_MONTHS)

RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
LAUNCH = pd.Timestamp("2002-03-15")
REAL_FROM = pd.Timestamp("2009-01-01")
SYNTH_TER = 0.0040          # assumed fund fee for the pre-launch synthetic era (estimate)
WHT = 0.85                  # Irish fund keeps 85% of US dividends
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def yahoo(name):
    d = pd.read_csv(RAW / "etf" / name, parse_dates=["date"]).set_index("date").sort_index()
    d.columns = [c.lower().replace(" ", "_") for c in d.columns]
    return d[~d.index.duplicated(keep="last")]


def asof(series, dates):
    """Latest value of `series` at or before each date (e.g. ECB fix on a holiday)."""
    return series.sort_index().reindex(dates, method="ffill")


def clean_ohlc(d):
    """Fix impossible bars: low must be <= open/close <= high."""
    d = d.copy()
    bad = (d["low"] > d[["open", "close"]].min(axis=1)) | (d["high"] < d[["open", "close"]].max(axis=1))
    d["low"] = d[["low", "open", "close"]].min(axis=1)
    d["high"] = d[["high", "open", "close"]].max(axis=1)
    return d, int(bad.sum())


def load_ishares(sheet):
    raw = pd.read_csv(PROC / "ishares" / f"{sheet}.csv", header=None, dtype=str)
    raw.columns = raw.iloc[0].str.strip()
    return raw.iloc[1:].reset_index(drop=True)


def main():
    fx = pd.read_csv(RAW / "fx" / "eurusd_ecb.csv", parse_dates=["date"]).set_index("date")["usd_per_eur"]
    spx = yahoo("GSPC.csv")[["open", "high", "low", "close"]]
    sptr = yahoo("SP500TR.csv")["close"]
    tnx = yahoo("../macro/TNX_yahoo.csv")["close"] if (RAW / "macro" / "TNX_yahoo.csv").exists() else None

    # ---------------------------------------------------------------- S&P quality
    section("S&P 500 daily bars (used for proxy open/high/low before 2009)")
    q = pd.DataFrame({
        "open_eq_prevclose": spx["open"] == spx["close"].shift(),
        "flat": spx["high"] == spx["low"]}).groupby(spx.index.year).mean().mul(100).round(1)
    print(q.loc[:2010].to_string())

    # ---------------------------------------------------------------- era 3: real IUSA
    iu = yahoo("IUSA.AS.csv")
    iu = iu[iu.index >= REAL_FROM]
    live = (iu["volume"] > 0) & (iu["high"] > iu["low"])
    real_start = live.idxmax()
    iu = iu[iu.index >= real_start][["open", "high", "low", "close", "volume"]]
    iu, n_bad = clean_ohlc(iu)
    flat_after = ((iu["high"] == iu["low"]).iloc[:250]).mean()
    section("ERA 3  real IUSA.AS")
    print(f"real data starts {real_start.date()} (2008 rows dropped: frozen, zero volume)")
    print(f"bars fixed (low/high outside open/close): {n_bad} | flat share first 250 days: {flat_after:.1%}")

    # ---------------------------------------------------------------- era 2: fund NAV
    st = load_ishares("Storico")
    nav = pd.DataFrame({
        "date": pd.to_datetime(st["Al"], format="%d/%m/%Y", errors="coerce"),
        "nav_usd": pd.to_numeric(st["NAV"], errors="coerce"),
        "fund_tr_idx": pd.to_numeric(st["Serie Fund Return"], errors="coerce"),
    }).dropna(subset=["date", "nav_usd"]).set_index("date").sort_index()
    nav = nav[~nav.index.duplicated(keep="last")]
    nav["nav_eur"] = nav["nav_usd"] / asof(fx, nav.index)
    section("ERA 2  iShares NAV")
    print(f"NAV rows {len(nav)}  {nav.index.min().date()} -> {nav.index.max().date()}  "
          f"launch NAV {nav['nav_usd'].iloc[0]} USD")

    # validation on overlap: real IUSA close vs NAV in EUR
    ov = pd.concat([iu["close"], nav["nav_eur"]], axis=1, join="inner").dropna()
    prem = ov["close"] / ov["nav_eur"] - 1
    r = ov.pct_change().dropna()
    wk = ov.resample("W-FRI").last().pct_change().dropna()
    print(f"overlap {len(ov)} days: IUSA vs NAV(EUR) premium median {prem.median():+.2%}, "
          f"5-95% [{prem.quantile(.05):+.2%}, {prem.quantile(.95):+.2%}]")
    print(f"return corr daily {r.iloc[:, 0].corr(r.iloc[:, 1]):.3f} | weekly {wk.iloc[:, 0].corr(wk.iloc[:, 1]):.3f}"
          "  (daily lower: NAV is struck at US close, IUSA closes 17:30 CET)")

    nav_era = nav[(nav.index >= LAUNCH) & (nav.index < real_start)].copy()
    ratios = (spx[["open", "high", "low"]].div(spx["close"], axis=0)).reindex(nav_era.index)
    for c in ["open", "high", "low"]:
        nav_era[c] = nav_era["nav_eur"] * ratios[c].fillna(1.0)
    nav_era["close"] = nav_era["nav_eur"]
    nav_era["proxy_flat"] = ratios["open"].isna()
    print(f"NAV-era days {len(nav_era)} | days with no S&P bar (flat proxy, no fills possible): "
          f"{int(nav_era['proxy_flat'].sum())}")

    # ---------------------------------------------------------------- era 1: synthetic
    s = spx[spx.index < LAUNCH].copy()
    launch_spx = spx["close"].asof(LAUNCH)
    launch_nav = nav["nav_usd"].asof(LAUNCH)
    yrs_to_launch = (LAUNCH - s.index).days / 365.25
    scale = launch_nav / launch_spx * np.exp(SYNTH_TER * yrs_to_launch)   # earlier = fee not yet charged
    fx_s = asof(fx, s.index)
    syn = pd.DataFrame(index=s.index)
    for c in ["open", "high", "low", "close"]:
        syn[c] = s[c] * scale / fx_s
    section("ERA 1  synthetic (S&P 500 in EUR, pre-launch)")
    print(f"{len(syn)} days {syn.index.min().date()} -> {syn.index.max().date()} | "
          f"S&P {launch_spx:.2f} at launch vs NAV {launch_nav} USD -> ratio 1/{launch_spx/launch_nav:.1f}")

    # ---------------------------------------------------------------- stitch
    cols = ["open", "high", "low", "close"]
    px = pd.concat([
        syn[cols].assign(era="synthetic", volume=np.nan),
        nav_era[cols].assign(era="nav", volume=np.nan),
        iu[cols + ["volume"]].assign(era="real"),
    ]).sort_index()
    px = px[~px.index.duplicated(keep="last")]
    px, n_fix = clean_ohlc(px)
    px["usd_per_eur"] = asof(fx, px.index)
    if tnx is not None:
        px["tnx"] = asof(tnx, px.index)

    section("STITCHED PRICE HISTORY")
    print(px.groupby("era").agg(first=("close", lambda x: x.index.min().date()),
                                last=("close", lambda x: x.index.max().date()),
                                days=("close", "size")).to_string())
    for d in [LAUNCH, real_start]:
        i = px.index.get_loc(d)
        a, b = px.iloc[i - 1], px.iloc[i]
        print(f"junction {px.index[i-1].date()} ({a.era}) {a.close:.3f} -> {d.date()} ({b.era}) "
              f"{b.close:.3f}  return {b.close/a.close-1:+.2%}")
    rets = px["close"].pct_change()
    print("largest daily moves (should be real crashes, not junk):")
    print(rets.abs().nlargest(8).round(4).to_string())

    # ---------------------------------------------------------------- distributions
    section("DISTRIBUTIONS (dividends)")
    di = load_ishares("Distribuzioni")
    dist = pd.DataFrame({
        "ex_date": pd.to_datetime(di["Data di godimento"], format="%d/%m/%Y", errors="coerce"),
        "pay_date": pd.to_datetime(di["Data del pagamento"], format="%d/%m/%Y", errors="coerce"),
        "amount_usd": pd.to_numeric(di["Distribuzione totale"], errors="coerce"),
    }).dropna().sort_values("ex_date")
    dist["usd_per_eur"] = asof(fx, pd.DatetimeIndex(dist["ex_date"])).values
    dist["amount_eur"] = dist["amount_usd"] / dist["usd_per_eur"]
    dist["source"] = "ishares"
    print(f"iShares: {len(dist)} payouts {dist.ex_date.min().date()} -> {dist.ex_date.max().date()}, "
          f"per year: {dist.groupby(dist.ex_date.dt.year).size().value_counts().to_dict()} (count: years)")
    print(f"pay date - ex date: median {int((dist.pay_date-dist.ex_date).dt.days.median())} days")

    # cross-check vs Yahoo EUR amounts (2015+)
    y = yahoo("IUSA.AS.csv")
    ydiv = y.loc[y["dividends"] > 0, "dividends"]
    m = pd.merge_asof(dist.set_index("ex_date").sort_index(), ydiv.rename("yahoo_eur").to_frame(),
                      left_index=True, right_index=True, direction="nearest",
                      tolerance=pd.Timedelta(days=3)).dropna(subset=["yahoo_eur"])
    diff = (m["amount_eur"] / m["yahoo_eur"] - 1)
    print(f"vs Yahoo EUR amounts on {len(m)} matches: median diff {diff.median():+.2%}, "
          f"max abs {diff.abs().max():.2%}  (small = FX rate date differences)")

    # synthetic dividends 1999 -> launch: from total-return vs price index
    tr = pd.concat([sptr.rename("tr"), spx["close"].rename("p")], axis=1).dropna()
    tr = tr[tr.index < LAUNCH]
    dy = (tr["tr"].pct_change() - tr["p"].pct_change()).clip(lower=0)      # daily dividend yield
    per_q = (dy * px["close"].reindex(tr.index)).groupby(tr.index.to_period("Q")).sum() * WHT
    syn_d = pd.DataFrame({
        "ex_date": [px.index[px.index.to_period("Q") == q].max() for q in per_q.index],
        "amount_eur": per_q.values})
    syn_d["pay_date"] = syn_d["ex_date"] + pd.Timedelta(days=14)
    syn_d["amount_usd"] = np.nan
    syn_d["usd_per_eur"] = np.nan
    syn_d["source"] = "synthetic"
    syn_d = syn_d[syn_d["ex_date"] < LAUNCH]
    dist = pd.concat([syn_d, dist], ignore_index=True).sort_values("ex_date")
    yld = dist.groupby(dist.ex_date.dt.year)["amount_eur"].sum() / px["close"].groupby(px.index.year).mean()
    print("annual payout yield (should be ~1-2%, net of 15% US tax):")
    print((yld * 100).round(2).loc[:2026].to_string())

    # ---------------------------------------------------------------- Acc fund (SXR8)
    section("SXR8 (accumulating version) - for the Dist vs Acc study")
    if (RAW / "etf" / "SXR8.DE.csv").exists():
        sx = yahoo("SXR8.DE.csv")
        live = (sx["volume"] > 0) & (sx["high"] > sx["low"])
        sx_start = live.idxmax()
        sx = sx[sx.index >= sx_start][["open", "high", "low", "close", "volume"]]
        sx, n_sx = clean_ohlc(sx)
        print(f"real SXR8 data from {sx_start.date()} (earlier rows frozen) | bars fixed {n_sx}")
        sx.to_parquet(PROC / "pit_sxr8.parquet")
        a, b = sx.index.min(), sx.index.max()
        yrs = (b - a).days / 365.25
        g_sx = (sx["close"].iloc[-1] / sx["close"].iloc[0]) ** (1 / yrs) - 1
        p0, p1 = px["close"].asof(a), px["close"].asof(b)
        g_pr = (p1 / p0) ** (1 / yrs) - 1
        print(f"{a.date()} -> {b.date()}: SXR8 {g_sx:.2%}/yr vs IUSA price only {g_pr:.2%}/yr "
              f"-> gap {g_sx-g_pr:.2%}/yr = dividends kept inside the Acc fund")

    # ---------------------------------------------------------------- Shiller
    section(f"SHILLER with {SHILLER_LAG_MONTHS}-month lag")
    sh = pd.read_parquet(PROC / "shiller_monthly.parquet")
    sh_av = shiller_available(sh[["E", "D", "CAPE", "GS10"]], px.index)
    last = sh_av.dropna(subset=["E"]).iloc[-1]
    print(f"on {px.index[-1].date()} the newest usable earnings month is "
          f"{last['shiller_month']:%Y-%m} (E={last['E']:.1f})")
    ex = sh_av.loc["2020-06-15":"2020-06-15"]
    if len(ex):
        print(f"example: on 2020-06-15 the strategy sees Shiller month {ex['shiller_month'].iloc[0]:%Y-%m}")

    # ---------------------------------------------------------------- features + test
    feats = known_at_open_features(px)
    section("LOOK-AHEAD TEST on the real stitched data")
    n = len(px)
    cut = [int(n * f) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
    fails = lookahead_check(known_at_open_features, px[["open", "high", "low", "close", "usd_per_eur"] +
                                                        (["tnx"] if "tnx" in px else [])], cut)
    print("scrambled everything from 5 cutoff dates onward (incl. that day's close/high/low);")
    print("checked every feature on and before the cutoff is unchanged ->",
          "PASS" if not fails else f"FAIL {fails}")

    # ---------------------------------------------------------------- write
    px.to_parquet(PROC / "pit_prices.parquet")
    feats.to_parquet(PROC / "pit_features.parquet")
    dist.to_parquet(PROC / "pit_distributions.parquet", index=False)
    sh_av.to_parquet(PROC / "pit_shiller.parquet")
    section("WROTE")
    for f in ["pit_prices", "pit_features", "pit_distributions", "pit_shiller", "pit_sxr8"]:
        p = PROC / f"{f}.parquet"
        if p.exists():
            print(f"  data/processed/{p.name}  {p.stat().st_size/1e3:.0f} KB")
    print("\nDONE. Paste everything back.")
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
