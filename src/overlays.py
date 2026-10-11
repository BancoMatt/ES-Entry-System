"""
src/overlays.py  -  study 10: do TREND (200-day EMA) or VALUE (Matt's formula) add anything?

An overlay wraps any policy (quarterly B&H, monthly DCA, a dip rule) and changes only HOW MUCH
each order spends. The inner policy keeps deciding WHEN.

  trend "above"  buy only if yesterday's close was above its 200-day EMA (uptrend); otherwise the
                 order is skipped and the cash waits for the next buy day that qualifies.
  trend "below"  the opposite: buy only in downtrends.
  value          every order x 1.5 when the index is cheap, x 1 in between, x 0.5 when expensive.
                 The half not spent when expensive goes into a "saved" pot; when cheap, the extra
                 half comes out of that pot (never more than is in it). Unspent cash waits.

VALUE SIGNAL (pre-registered §11, Matt's formula), on the S&P 500 in USD, known at the decision:
  EPS  = Shiller earnings, 6-month publication lag (src/pit.py)
  g    = 10-year EPS growth per year, capped to [0%, 10%]
  M    = median P/E of the 120 months before (same lag)
  P    = S&P 500 close of the previous trading day
  d    = dividend yield = Shiller dividends (lagged) / P
  e    = (EPS x (1+g)^10 x M / P)^(1/10) - 1 + d        expected yearly return over 10 years
  r    = 10-year US yield (previous close) + 5%          required return
  cheap if e - r >= +1%  -> x1.5 | expensive if e - r <= -1% -> x0.5 | otherwise x1
"""
import numpy as np
import pandas as pd

from src.engine import MOC, Limit, Policy

CHEAP, EXPENSIVE = 0.01, -0.01


def value_signal(spx_close, shiller_monthly, pit_shiller, tnx_close=None, n=10):
    """Daily frame (index = trading days of pit_shiller): e, r, e_minus_r, mult. All point-in-time."""
    sm = shiller_monthly[["P", "E", "D"]].astype(float)
    pe = sm["P"] / sm["E"]
    m_pe = pe.rolling(120, min_periods=120).median()
    g = ((sm["E"] / sm["E"].shift(120)) ** (1 / 10) - 1).clip(0.0, 0.10)
    per_month = pd.DataFrame({"M": m_pe, "g": g, "E": sm["E"], "D": sm["D"]})
    days = pit_shiller.index
    lagged = per_month.reindex(pit_shiller["shiller_month"].to_numpy())
    lagged.index = days
    p = spx_close.sort_index().shift(1).reindex(days, method="ffill")            # yesterday's close
    if tnx_close is not None:
        y = tnx_close.sort_index().shift(1).reindex(days, method="ffill") / 100
    else:
        y = pit_shiller["GS10"] / 100
    eps_n = lagged["E"] * (1 + lagged["g"]) ** n
    e = (eps_n * lagged["M"] / p) ** (1 / n) - 1 + lagged["D"] / p
    r = y + 0.05
    out = pd.DataFrame({"e": e, "r": r, "e_minus_r": e - r, "spx_prev": p, "M": lagged["M"], "g": lagged["g"]})
    out["mult"] = np.where(out["e_minus_r"] >= CHEAP, 1.5, np.where(out["e_minus_r"] <= EXPENSIVE, 0.5, 1.0))
    out.loc[out["e_minus_r"].isna(), "mult"] = 1.0
    return out


class _View:
    """What the inner policy sees: the real engine, minus the cash set aside by the overlay."""
    def __init__(self, eng, ov):
        object.__setattr__(self, "_eng", eng)
        object.__setattr__(self, "_ov", ov)

    def __getattr__(self, k):
        return getattr(self._eng, k)

    def __setattr__(self, k, v):
        setattr(self._eng, k, v)

    @property
    def cash(self):
        return max(self._eng.cash - self._ov.saved, 0.0)


class Overlay(Policy):
    def __init__(self, inner, kind, trend_ok=None, mult=None):
        """kind: 'trend_above' | 'trend_below' | 'value'. trend_ok / mult: Series by date."""
        self.inner, self.kind = inner, kind
        self.trend_ok, self.mult_s = trend_ok, mult
        self.name = f"{inner.name}+{kind}"

    def setup(self, eng):
        self.saved = 0.0
        self.view = _View(eng, self)
        if self.trend_ok is not None:
            self.ok = self.trend_ok.reindex(eng.dates).fillna(False).to_numpy(bool)
        if self.mult_s is not None:
            self.m = self.mult_s.reindex(eng.dates).fillna(1.0).to_numpy(float)
        self.inner.setup(self.view)

    def _adjust(self, eng, i, orders):
        if not orders:
            return orders
        if self.kind.startswith("trend"):
            return orders if self.ok[i] else []
        m = self.m[i]
        out = []
        for od in orders:
            b = min(od.budget, max(eng.cash - self.saved, 0.0))     # never touch the saved pot by accident
            if m < 1:
                keep = b * (1 - m)
                self.saved += keep
                b -= keep
            elif m > 1:
                extra = min(self.saved, b * (m - 1))
                self.saved -= extra
                b += extra
            od.budget = min(b, eng.cash)
            out.append(od)
        return out

    def on_open(self, eng, i):
        return self._adjust(eng, i, self.inner.on_open(self.view, i))

    def on_close(self, eng, i):
        return self._adjust(eng, i, self.inner.on_close(self.view, i))

    def after_close(self, eng, i, filled):
        self.inner.after_close(self.view, i, filled)
        self.saved = min(self.saved, eng.cash)
