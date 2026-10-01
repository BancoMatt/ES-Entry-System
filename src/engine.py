"""
src/engine.py  -  plays the account forward one trading day at a time.

The engine is the ACCOUNTANT. A Policy (benchmark or strategy) is the DECISION-MAKER.
The engine never decides what to buy; the policy never touches cash directly.

ONE TRADING DAY, in the order things really happen
  1. Morning   money arrives: contributions (first trading day of the period) and
               dividends whose PAY date is today (26% tax taken). On an EX date we note
               how many shares were held this morning - that's who gets paid later.
  2. Open      policy.on_open()  -> limit orders  (E1). Filled if the day's low reaches
               the trigger; price = min(open, limit). No spread (you get your price).
  3. Close     policy.on_close() -> market-on-close orders (benchmarks, E2, deadlines).
               Price = close x (1 + half-spread).
  4. Evening   policy.after_close() -> policy updates its own state using today's close.
               The account is valued: shares x close + cash.

Cash is kept in LOTS (date it arrived, amount), spent oldest-first. That lets a policy
ask "how old is my oldest cash?" for the deadline rule.

Every euro is accounted for:  contributions + net dividends = cash + spent on shares + fees
(tests/test_engine.py checks this on every run).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.costs import CostModel, IBKR_TIERED


# ------------------------------------------------------------------ contributions
def contribution_schedule(dates: pd.DatetimeIndex, start, end, per_year=2000.0, freq="Q"):
    """First trading day of each year (A), quarter (Q) or month (M), inside [start, end]."""
    d = dates[(dates >= start) & (dates <= end)]
    s = pd.Series(d, index=d)
    period = {"A": d.to_period("Y"), "Q": d.to_period("Q"), "M": d.to_period("M")}[freq]
    firsts = s.groupby(period).min()
    amount = per_year / {"A": 1, "Q": 4, "M": 12}[freq]
    return {pd.Timestamp(x): amount for x in firsts.values}


# ------------------------------------------------------------------ orders
@dataclass
class Limit:
    limit: float                 # price placed at the open
    budget: float                # euros this order may spend (fee included)
    trade_through: float = 0.0   # 0.0005 = the low must go 0.05% BELOW the limit to count
    tag: str = "limit"


@dataclass
class MOC:
    budget: float
    tag: str = "moc"


class Policy:
    """Base class. Benchmarks and strategies override what they need."""
    name = "policy"

    def setup(self, eng):                 # called once before day 1
        pass

    def on_open(self, eng, i):            # -> list[Limit]
        return []

    def on_close(self, eng, i):           # -> list[MOC]
        return []

    def after_close(self, eng, i, filled):  # filled = list of today's trades
        pass


# ------------------------------------------------------------------ engine
@dataclass
class Result:
    name: str
    daily: pd.DataFrame
    trades: pd.DataFrame
    flows: list
    totals: dict = field(default_factory=dict)


class Engine:
    def __init__(self, px, dist, start, end, policy, cost: CostModel = IBKR_TIERED,
                 per_year=2000.0, freq="Q"):
        self.px_all = px
        self.dates = px.index[(px.index >= start) & (px.index <= end)]
        self.px = px.loc[self.dates]
        self.o, self.h, self.l, self.c = (self.px[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.policy, self.cost = policy, cost
        self.contrib = contribution_schedule(self.dates, start, end, per_year, freq)
        self.per_year, self.freq = per_year, freq
        # dividends: map ex-date and pay-date onto trading days inside the period
        self.ex_days, self.pay_amt = {}, {}
        for r in dist.itertuples():
            ex_i = self.dates.searchsorted(pd.Timestamp(r.ex_date))
            pay_i = self.dates.searchsorted(pd.Timestamp(r.pay_date))
            if ex_i < len(self.dates) and pay_i < len(self.dates) and self.dates[ex_i] >= start:
                self.ex_days.setdefault(ex_i, []).append((pay_i, float(r.amount_eur)))
        # state
        self.lots = []                # [arrival_date, euros]
        self.shares = 0.0
        self.cost_basis = 0.0
        self.fees = self.div_gross = self.div_tax = self.spent = 0.0
        self.contributed = 0.0
        self.pending_div = {}         # pay_i -> gross euros
        self.wait_weighted = 0.0      # sum of (euros invested x days they waited in cash)
        self.wait_euros = 0.0
        self.trades, self.flows = [], []

    # ---------------- cash helpers
    @property
    def cash(self):
        return sum(a for _, a in self.lots)

    def oldest_cash_age(self, i):
        return (self.dates[i] - self.lots[0][0]).days if self.lots else 0

    def _add_cash(self, date, amount):
        if amount > 0:
            self.lots.append([date, amount])

    def _spend(self, amount, today=None):
        tol = 1e-9 * max(1.0, amount)          # float rounding scales with the amount
        amount = round(amount, 10)
        while amount > tol and self.lots:
            take = min(amount, self.lots[0][1])
            if today is not None:
                self.wait_weighted += take * (today - self.lots[0][0]).days
                self.wait_euros += take
            self.lots[0][1] -= take
            amount -= take
            if self.lots[0][1] <= tol:
                self.lots.pop(0)
        if amount > 1e-6 * max(1.0, tol * 1e9):
            raise RuntimeError("spent more cash than available")

    # ---------------- trading
    def _buy(self, i, budget, price, kind, tag):
        budget = min(budget, self.cash)
        n = self.cost.affordable(budget, price)
        if n <= 0:
            return None
        value = n * price
        fee = self.cost.fee(value)
        self._spend(value + fee, self.dates[i])
        self.shares += n
        self.cost_basis += value + fee
        self.fees += fee
        self.spent += value
        t = {"date": self.dates[i], "kind": kind, "tag": tag, "shares": n, "price": price,
             "value": value, "fee": fee}
        self.trades.append(t)
        return t

    # ---------------- main loop
    def run(self) -> Result:
        self.policy.setup(self)
        rows = []
        for i, d in enumerate(self.dates):
            # 1. morning
            if d in self.contrib:
                amt = self.contrib[d]
                self._add_cash(d, amt)
                self.contributed += amt
                self.flows.append((d, -amt))
            for pay_i, amt in self.ex_days.get(i, []):
                self.pending_div[pay_i] = self.pending_div.get(pay_i, 0.0) + self.shares * amt
            if i in self.pending_div:
                gross = self.pending_div.pop(i)
                tax = gross * self.cost.div_tax
                self.div_gross += gross
                self.div_tax += tax
                self._add_cash(d, gross - tax)
            filled = []
            # 2. open: limit orders
            for od in self.policy.on_open(self, i) or []:
                trigger = od.limit * (1 - od.trade_through)
                if self.l[i] <= trigger:
                    t = self._buy(i, od.budget, min(self.o[i], od.limit), "limit", od.tag)
                    if t:
                        filled.append(t)
            # 3. close: market-on-close
            for od in self.policy.on_close(self, i) or []:
                t = self._buy(i, od.budget, self.c[i] * (1 + self.cost.half_spread), "moc", od.tag)
                if t:
                    filled.append(t)
            # 4. evening
            self.policy.after_close(self, i, filled)
            cash = self.cash
            rows.append((d, self.shares, cash, self.shares * self.c[i] + cash, self.contributed))
        # cash never invested counts as having waited until the last day
        for d0, a in self.lots:
            self.wait_weighted += a * (self.dates[-1] - d0).days
            self.wait_euros += a
        daily = pd.DataFrame(rows, columns=["date", "shares", "cash", "value", "contributed"]).set_index("date")
        res = Result(self.policy.name, daily, pd.DataFrame(self.trades), list(self.flows))
        res.totals = {"contributed": self.contributed, "div_gross": self.div_gross,
                      "div_tax": self.div_tax, "fees": self.fees, "spent_on_shares": self.spent,
                      "cost_basis": self.cost_basis, "cash_end": self.cash,
                      "shares_end": self.shares, "value_end": float(daily["value"].iloc[-1]),
                      "avg_wait_days": self.wait_weighted / self.wait_euros if self.wait_euros else 0.0,
                      "end_date": self.dates[-1]}
        return res


def run(px, dist, start, end, policy, **kw) -> Result:
    return Engine(px, dist, start, end, policy, **kw).run()
