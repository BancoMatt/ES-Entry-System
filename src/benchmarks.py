"""
src/benchmarks.py  -  the two things every strategy has to beat.

BuyHold      invests ALL available cash at the close of every contribution day.
             Dividends that arrived in between are invested with the next contribution
             (a real person doesn't place a EUR 1.55-fee order for a EUR 20 dividend).
             With quarterly money this is also "quarterly DCA".

MonthlyDCA   buys on day `d` of every month (first trading day on/after d).
             Money still arrives quarterly, so each buy spends an equal share of what's
             left until the next contribution: 1st buy = cash/3, 2nd = cash/2, 3rd = all.
             That also sweeps up rounding leftovers and dividends.

PerfectTimingCeiling   NOT A STRATEGY. Uses hindsight: invests each quarter's cash at that
             quarter's LOWEST close. No rule can do better by timing within the quarter,
             so it's the ceiling for "how much edge was even possible". Descriptive only,
             never selectable (logged in notes/decisions.md).

--- added for study 06 (funding) and the calendar curiosity ---------------------------
SpreadBuys   splits whatever cash is there over the buy days left before the next
             contribution: first trading day of each quarter ("Q") or month ("M").
             With EUR 2,000 arriving in January, "Q" = 4 buys of ~EUR 500.
InvestOnDays invests ALL cash on a fixed list of trading-day indices (e.g. the k-th
             trading day of every year). Money that arrives earlier just waits.
nth_day_of_year / extreme_day_of_year   helpers that build those day lists.
"""
import numpy as np
import pandas as pd

from src.engine import MOC, Policy


class BuyHold(Policy):
    name = "buy_hold"

    def setup(self, eng):
        self.buy_days = {eng.dates.get_loc(d) for d in eng.contrib}

    def on_close(self, eng, i):
        return [MOC(eng.cash, "bh")] if i in self.buy_days else []


class MonthlyDCA(Policy):
    def __init__(self, day=1):
        self.day = day
        self.name = f"dca_monthly_d{day:02d}"

    def setup(self, eng):
        dates = eng.dates
        s = pd.Series(range(len(dates)), index=dates)
        buys = []
        for _, grp in s.groupby(dates.to_period("M")):
            on_or_after = grp[grp.index.day >= self.day]
            if len(on_or_after):
                buys.append(int(on_or_after.iloc[0]))
        contrib_i = sorted(dates.get_loc(d) for d in eng.contrib)
        # for each buy: how many buys remain (incl. this one) before the next contribution
        self.split = {}
        for b in buys:
            nxt = next((c for c in contrib_i if c > b), len(dates))
            self.split[b] = sum(1 for x in buys if b <= x < nxt)

    def on_close(self, eng, i):
        k = self.split.get(i)
        return [MOC(eng.cash / k, "dca")] if k else []


class PerfectTimingCeiling(Policy):
    name = "ceiling_perfect_timing"          # HINDSIGHT - reference line only

    def setup(self, eng):
        contrib = sorted(eng.dates.get_loc(d) for d in eng.contrib)
        bounds = contrib + [len(eng.dates)]
        self.buy_days = set()
        for a, b in zip(bounds[:-1], bounds[1:]):
            self.buy_days.add(a + int(np.argmin(eng.c[a:b])))

    def on_close(self, eng, i):
        return [MOC(eng.cash, "ceiling")] if i in self.buy_days else []


# ------------------------------------------------------------------ study 06 + calendar
def nth_day_of_year(dates, k):
    """Index of the k-th trading day (1 = first) of every calendar year in `dates`."""
    s = pd.Series(range(len(dates)), index=dates)
    out = []
    for _, grp in s.groupby(dates.year):
        if len(grp) >= k:
            out.append(int(grp.iloc[k - 1]))
    return out


def extreme_day_of_year(dates, closes, how="min"):
    """HINDSIGHT. Index of the lowest ("min") or highest ("max") close of every year."""
    s = pd.Series(closes, index=dates)
    out = []
    base = pd.Series(range(len(dates)), index=dates)
    for _, grp in s.groupby(dates.year):
        day = grp.idxmin() if how == "min" else grp.idxmax()
        out.append(int(base.loc[day]))
    return out


class SpreadBuys(Policy):
    def __init__(self, freq="Q"):
        self.freq = freq
        self.name = f"spread_{freq}"

    def setup(self, eng):
        dates = eng.dates
        s = pd.Series(range(len(dates)), index=dates)
        buys = [int(g.iloc[0]) for _, g in s.groupby(dates.to_period(self.freq))]
        contrib_i = sorted(dates.get_loc(d) for d in eng.contrib)
        self.split = {}
        for b in buys:
            nxt = next((c for c in contrib_i if c > b), len(dates))
            self.split[b] = sum(1 for x in buys if b <= x < nxt)

    def on_close(self, eng, i):
        k = self.split.get(i)
        return [MOC(eng.cash / k, "spread")] if k else []


class InvestOnDays(Policy):
    """`days` = list of trading-day indices, or a function eng -> list (resolved at setup)."""
    def __init__(self, days, name="invest_on_days"):
        self._days = days
        self.name = name

    def setup(self, eng):
        d = self._days(eng) if callable(self._days) else self._days
        self.days = set(int(x) for x in d)

    def on_close(self, eng, i):
        return [MOC(eng.cash, "day")] if i in self.days else []
