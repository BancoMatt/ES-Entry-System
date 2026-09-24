"""
src/signals.py  -  entry rules. Reused by the frequency study AND (later) the engine.

Every rule places a LIMIT BUY at the open of day t, at a price computed only from
information known at the open (src/pit.py). The order fills if that day's low
reaches the limit; if the day OPENS below the limit, it fills at the open (better).

REFERENCE PRICE (what the drop is measured from)
  open        today's open                  (needs a real open: not before 2007 here)
  prev_close  yesterday's close
  high_Nd     highest close of the previous N days   (drawdown from a recent high)

THRESHOLD (how big the drop must be)
  fixed       x = a fixed %                     e.g. 1.25%
  vol         x = k * recent daily volatility   (k * vol_20d) -> adapts to calm/wild years

RE-ENTRY RULE (only matters for high_Nd: how many buys inside one drop)
  A  one entry per drop. Re-arms only after a close at a NEW N-day high.
  B  ladder: buy at -1x, -2x, -3x ... below the high where the drop started;
     each level once per drop; everything resets at a new N-day high.
  C  cooldown: may buy again whenever the condition holds and >= `cooldown` days
     passed since the last buy.
For open / prev_close references every day stands alone (one possible entry per day).

State updates (re-arming) use the day's CLOSE and take effect the NEXT day.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

OPEN_RELIABLE_FROM = pd.Timestamp("2007-01-01")   # Yahoo S&P opens are fake before this


@dataclass(frozen=True)
class Rule:
    ref: str                 # "open" | "prev_close" | "high_5d" | "high_20d" ...
    thr_kind: str            # "fixed" | "vol"
    thr: float               # fixed: 0.0125 = 1.25% ; vol: k multiplier (e.g. 2.0)
    reentry: str = "-"       # "A" | "B" | "C" | "-" (daily refs)
    cooldown: int = 0        # for C
    ladder_max: int = 5      # for B

    @property
    def name(self):
        t = f"{self.thr:.2%}" if self.thr_kind == "fixed" else f"{self.thr:g}xvol"
        extra = f"C{self.cooldown}" if self.reentry == "C" else self.reentry
        return f"{self.ref}|{t}|{extra}"


def entries(px: pd.DataFrame, feat: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    """
    Returns one row per day: n (number of limit levels filled that day, 0..ladder_max),
    limit (first limit price in force at the open, NaN if no order), fill (price paid
    for the first fill, NaN if none). Days where the rule can't be evaluated -> n = NaN.
    """
    o, lo, c = (px[k].to_numpy(float) for k in ("open", "low", "close"))
    idx = px.index
    if rule.thr_kind == "fixed":
        x = np.full(len(px), rule.thr)
    else:
        x = rule.thr * feat["vol_20d"].to_numpy(float)

    n = np.zeros(len(px))
    limit = np.full(len(px), np.nan)
    fill = np.full(len(px), np.nan)

    if rule.ref in ("open", "prev_close"):
        ref = o if rule.ref == "open" else feat["prev_close"].to_numpy(float)
        lvl = ref * (1 - x)
        hit = lo <= lvl
        n = hit.astype(float)
        limit = lvl
        fill = np.where(hit, np.minimum(o, lvl), np.nan)
        bad = np.isnan(lvl)
        if rule.ref == "open":
            bad |= idx < OPEN_RELIABLE_FROM
        n[bad] = np.nan
        return pd.DataFrame({"n": n, "limit": limit, "fill": fill}, index=idx)

    H = feat[rule.ref].to_numpy(float)              # prior N-day high, known at open
    armed, last_fire = True, -10**9
    anchor, x_ep, fired = np.nan, np.nan, 0         # ladder episode state
    for i in range(len(px)):
        if np.isnan(H[i]) or np.isnan(x[i]):
            n[i] = np.nan
            continue
        if rule.reentry == "A":
            if armed:
                limit[i] = H[i] * (1 - x[i])
                if lo[i] <= limit[i]:
                    n[i], fill[i], armed = 1, min(o[i], limit[i]), False
        elif rule.reentry == "C":
            if i - last_fire >= rule.cooldown:
                limit[i] = H[i] * (1 - x[i])
                if lo[i] <= limit[i]:
                    n[i], fill[i], last_fire = 1, min(o[i], limit[i]), i
        elif rule.reentry == "B":
            a, xe = (H[i], x[i]) if fired == 0 else (anchor, x_ep)
            if fired < rule.ladder_max:
                limit[i] = a * (1 - (fired + 1) * xe)
            k = fired
            while k < rule.ladder_max and lo[i] <= a * (1 - (k + 1) * xe):
                k += 1
            if k > fired:
                if fired == 0:
                    anchor, x_ep = a, xe
                n[i], fill[i], fired = k - fired, min(o[i], limit[i]), k
        else:
            raise ValueError(rule.reentry)
        # end of day: a close at a new N-day high re-arms everything (effective tomorrow)
        if c[i] >= H[i]:
            armed, fired = True, 0
    return pd.DataFrame({"n": n, "limit": limit, "fill": fill}, index=idx)


def default_grid():
    """The candidate rules for the frequency study (to be frozen in the pre-registration)."""
    fixed = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    vol_k = [1.0, 1.5, 2.0, 2.5, 3.0]
    rules = []
    for ref in ("open", "prev_close"):
        rules += [Rule(ref, "fixed", t) for t in fixed]
        rules += [Rule(ref, "vol", k) for k in vol_k]
    for ref in ("high_5d", "high_10d", "high_20d", "high_50d"):
        for kind, ths in (("fixed", fixed), ("vol", vol_k)):
            for t in ths:
                rules += [Rule(ref, kind, t, "A"), Rule(ref, kind, t, "B"),
                          Rule(ref, kind, t, "C", cooldown=5), Rule(ref, kind, t, "C", cooldown=20)]
    return rules
