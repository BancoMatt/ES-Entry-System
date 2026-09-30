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
