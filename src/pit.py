"""
src/pit.py  -  point-in-time rules, in ONE place.

Every piece of information has a moment when it becomes known. Strategies may
only use information that was known at the moment they decide. For a daily
limit-order strategy the decision moment is THE OPEN of day t, so:

  known at the open of day t          NOT known at the open of day t
  --------------------------          ------------------------------
  all closes up to day t-1            today's close / high / low
  today's open                        anything later than today
  FX fixes up to day t-1              today's ECB fix (published ~16:00 CET)
  10y yield up to day t-1
  Shiller data older than the lag     Shiller months inside the lag

Today's low/high are used ONLY to check whether an order that was already
placed at the open got filled - never to decide anything.

The functions below build "known at open" features, and `lookahead_check`
proves it: it scrambles the data from a cutoff onward and verifies that no
feature before the cutoff changed.
"""
import numpy as np
import pandas as pd

HIGH_WINDOWS = (5, 10, 20, 50, 100, 250)   # candidate "recent high" look-backs (days)
VOL_WINDOW = 20
EMA_SPAN = 200
SHILLER_LAG_MONTHS = 6                     # fixed up front, see notes/decisions.md


def known_at_open_features(px: pd.DataFrame) -> pd.DataFrame:
    """
    px: daily frame indexed by date with columns open, close (+ optional
        usd_per_eur, tnx). Returns features usable at the OPEN of each day.
    Rule: everything derived from closes is shifted by one day.
    """
    c = px["close"]
    f = pd.DataFrame(index=px.index)
    f["open_today"] = px["open"]                      # known at the open by definition
    f["prev_close"] = c.shift(1)
    for n in HIGH_WINDOWS:
        f[f"high_{n}d"] = c.rolling(n, min_periods=n).max().shift(1)
    f[f"vol_{VOL_WINDOW}d"] = c.pct_change().rolling(VOL_WINDOW).std().shift(1)
    f[f"ema_{EMA_SPAN}"] = c.ewm(span=EMA_SPAN, adjust=False).mean().shift(1)
    if "usd_per_eur" in px:
        f["fx_prev"] = px["usd_per_eur"].shift(1)
    if "tnx" in px:
        f["tnx_prev"] = px["tnx"].shift(1)
    return f


def shiller_available(shiller: pd.DataFrame, dates: pd.DatetimeIndex,
                      lag_months: int = SHILLER_LAG_MONTHS) -> pd.DataFrame:
    """
    For each trading date, attach the latest Shiller month that was AVAILABLE.
    Month m (e.g. 2026-03) becomes usable on the 1st day of m + 1 + lag.
    With lag 6: March 2026 data is usable from 1 Oct 2026.
    """
    s = shiller.copy()
    s["available_from"] = s.index + pd.DateOffset(months=1 + lag_months)
    s["shiller_month"] = s.index
    s = s.set_index("available_from").sort_index()
    out = s.reindex(dates, method="ffill")
    out.index.name = "date"
    return out


def lookahead_check(feature_fn, px: pd.DataFrame, cutoffs, seed: int = 0) -> list:
    """
    Scramble every row from `cutoff` onward - INCLUDING the cutoff day's
    close/high/low (not its open, which is legitimately known) - and check that
    all features on rows <= cutoff are unchanged. Returns a list of failures.
    """
    rng = np.random.default_rng(seed)
    base = feature_fn(px)
    failures = []
    for k in cutoffs:
        p = px.copy()
        noise_cols = [c for c in p.columns if c != "open"]
        for col in noise_cols:
            vals = p[col].to_numpy(dtype=float, copy=True)
            vals[k:] *= rng.uniform(0.5, 1.5, size=len(vals) - k)
            p[col] = vals
        o = p["open"].to_numpy(dtype=float, copy=True)          # future opens scrambled too
        o[k + 1:] *= rng.uniform(0.5, 1.5, size=len(o) - k - 1)
        p["open"] = o
        test = feature_fn(p)
        a, b = base.iloc[: k + 1], test.iloc[: k + 1]
        for col in base.columns:
            same = np.isclose(a[col].to_numpy(float), b[col].to_numpy(float), equal_nan=True)
            if not same.all():
                failures.append((k, col, int((~same).sum())))
    return failures
