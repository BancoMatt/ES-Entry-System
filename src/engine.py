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

Added for study 11 (crash playbook), all OFF unless a policy uses them:
  SELLS    Sell orders at the close (price = close x (1 - half-spread)), fee charged, 26% tax
           on the realised gain (average cost). Losses give no tax credit (simplification).
  DEBT     eng.borrow(amount, i) puts borrowed cash in the account; interest accrues every
           day at `debt_rate` (annual, a Series by date); eng.repay(amount, i) pays it back
           from cash; with eng.repay_from_contrib = True each contribution repays debt first.
           Salary flows (XIRR) never change; the final value is NET of any debt left.
  Full books:  contributions kept + borrowed + net dividends + sale proceeds (after fee, tax)
             = cash + spent on shares + buy fees + repaid from cash      (check_books())
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


@dataclass
class Sell:
    fraction: float = 1.0        # share of the holding to sell (1.0 = everything)
    amount: float = None         # OR: euros wanted after fee and tax (overrides fraction)
    tag: str = "sell"


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
    sells: object = None


class Engine:
    def __init__(self, px, dist, start, end, policy, cost: CostModel = IBKR_TIERED,
                 per_year=2000.0, freq="Q", debt_rate=None):
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
        # study 11 extensions (inactive unless used)
        self.sells = []
        self.sell_value = self.sell_fees = self.cgt_paid = 0.0
        self.debt = self.borrowed = self.interest = 0.0
        self.repaid_from_contrib = self.repaid_from_cash = 0.0
        self.repay_from_contrib = False
        self.debt_rate = None
        if debt_rate is not None:
            self.debt_rate = debt_rate.sort_index().reindex(self.dates, method="ffill").fillna(0.0).to_numpy(float)

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

    # ---------------- debt
    def borrow(self, amount, i):
        if amount > 0:
            self._add_cash(self.dates[i], amount)
            self.debt += amount
            self.borrowed += amount

    def repay(self, amount, i):
        amt = min(amount, self.cash, self.debt)
        if amt > 1e-9:
            self._spend(amt)
            self.debt -= amt
            self.repaid_from_cash += amt
        return amt

    def check_books(self):
        """Residual of the full accounting identity (should be ~0)."""
        left = (self.contributed - self.repaid_from_contrib + self.borrowed + self.div_gross - self.div_tax
                + self.sell_value - self.sell_fees - self.cgt_paid)
        right = self.cash + self.spent + self.fees + self.repaid_from_cash
        return left - right

    # ---------------- trading
    def _sell(self, i, od):
        if self.shares <= 0:
            return None
        price = self.c[i] * (1 - self.cost.half_spread)
        if od.amount is not None:
            avg = self.cost_basis / self.shares
            net_per_share = price - self.cost.cgt * max(price - avg, 0.0)
            n = (od.amount + self.cost.fee(od.amount)) / net_per_share
            n = n if self.cost.fractional else float(np.ceil(n - 1e-9))
        else:
            n = self.shares * od.fraction
            n = n if self.cost.fractional or od.fraction >= 1 else float(np.floor(n))
        n = min(n, self.shares)
        if n <= 0:
            return None
        value = n * price
        fee = self.cost.fee(value)
        basis = self.cost_basis * n / self.shares
        tax = self.cost.cgt * max(value - fee - basis, 0.0)
        self.shares -= n
        self.cost_basis -= basis
        self._add_cash(self.dates[i], value - fee - tax)
        self.sell_value += value
        self.sell_fees += fee
        self.cgt_paid += tax
        t = {"date": self.dates[i], "kind": "sell", "tag": od.tag, "shares": n, "price": price,
             "value": value, "fee": fee, "tax": tax}
        self.sells.append(t)
        return t

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
            # 0. interest on any debt since the last trading day
            if self.debt > 0 and self.debt_rate is not None and i > 0:
                acc = self.debt * self.debt_rate[i] * (d - self.dates[i - 1]).days / 365
                self.debt += acc
                self.interest += acc
            # 1. morning
            if d in self.contrib:
                amt = self.contrib[d]
                rep = min(amt, self.debt) if self.repay_from_contrib else 0.0
                self.debt -= rep
                self.repaid_from_contrib += rep
                self._add_cash(d, amt - rep)
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
            # 3. close: sells first (their cash can be reused), then market-on-close buys
            orders = self.policy.on_close(self, i) or []
            for od in [o for o in orders if isinstance(o, Sell)]:
                t = self._sell(i, od)
                if t:
                    filled.append(t)
            for od in [o for o in orders if not isinstance(o, Sell)]:
                t = self._buy(i, od.budget, self.c[i] * (1 + self.cost.half_spread), "moc", od.tag)
                if t:
                    filled.append(t)
            # 4. evening
            self.policy.after_close(self, i, filled)
            cash = self.cash
            rows.append((d, self.shares, cash, self.shares * self.c[i] + cash, self.contributed, self.debt))
        # cash never invested counts as having waited until the last day
        for d0, a in self.lots:
            self.wait_weighted += a * (self.dates[-1] - d0).days
            self.wait_euros += a
        daily = pd.DataFrame(rows, columns=["date", "shares", "cash", "value", "contributed", "debt"]).set_index("date")
        res = Result(self.policy.name, daily, pd.DataFrame(self.trades), list(self.flows))
        res.sells = pd.DataFrame(self.sells)
        res.totals = {"contributed": self.contributed, "div_gross": self.div_gross,
                      "div_tax": self.div_tax, "fees": self.fees, "spent_on_shares": self.spent,
                      "cost_basis": self.cost_basis, "cash_end": self.cash,
                      "shares_end": self.shares, "value_end": float(daily["value"].iloc[-1]),
                      "avg_wait_days": self.wait_weighted / self.wait_euros if self.wait_euros else 0.0,
                      "end_date": self.dates[-1],
                      "debt_end": self.debt, "borrowed": self.borrowed, "interest": self.interest,
                      "repaid_from_contrib": self.repaid_from_contrib, "repaid_from_cash": self.repaid_from_cash,
                      "sell_value": self.sell_value, "sell_fees": self.sell_fees, "cgt_paid": self.cgt_paid,
                      "books_residual": self.check_books()}
        return res


def run(px, dist, start, end, policy, **kw) -> Result:
    return Engine(px, dist, start, end, policy, **kw).run()
