"""
tests/test_engine_debt_sell.py  -  the study 11 additions to the engine: sells, debt, interest.
Hand-checkable toy prices; zero fees and fractional shares unless the test is about them.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.costs import IBKR_TIERED, ZERO_FEES, with_fractional   # noqa: E402
from src.engine import MOC, Engine, Policy, Sell, run           # noqa: E402
from src.metrics import summarize                               # noqa: E402

FREE = with_fractional(ZERO_FEES)
NO_DIST = pd.DataFrame(columns=["ex_date", "pay_date", "amount_eur"])


def step_prices(p1=50.0, p2=60.0, start="2010-01-01", n=520, switch=260):
    idx = pd.bdate_range(start, periods=n)
    c = np.where(np.arange(n) < switch, p1, p2).astype(float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx)


class BuyThenSell(Policy):
    def __init__(self, sell_day, fraction=1.0, amount=None):
        self.sell_day, self.fraction, self.amount = sell_day, fraction, amount

    def setup(self, eng):
        self.buy_days = {eng.dates.get_loc(d) for d in eng.contrib}

    def on_close(self, eng, i):
        out = []
        if i == self.sell_day:
            out.append(Sell(fraction=self.fraction, amount=self.amount))
        if i in self.buy_days and i < self.sell_day:
            out.append(MOC(eng.cash))
        return out


def test_sell_all_pays_tax_on_gain_and_books_balance():
    px = step_prices()
    eng = Engine(px, NO_DIST, px.index[0], px.index[-1], BuyThenSell(300), cost=FREE, per_year=2000, freq="A")
    res = eng.run()
    t = res.totals
    assert t["shares_end"] == pytest.approx(0.0)
    s = res.sells.iloc[0]
    # Jan 2010: 2,000 at 50 = 40 sh | Jan 2011 (after the switch to 60): 2,000 at 60 = 33.33 sh
    shares = 40 + 2000 / 60
    assert s["value"] == pytest.approx(shares * 60)
    assert s["tax"] == pytest.approx(0.26 * (shares * 60 - 4000.0))
    assert abs(t["books_residual"]) < 1e-6


def test_sell_amount_raises_at_least_the_amount():
    px = step_prices()
    pol = BuyThenSell(300, amount=1000.0)
    eng = Engine(px, NO_DIST, px.index[0], px.index[-1], pol, cost=FREE, per_year=2000, freq="A")
    res = eng.run()
    s = res.sells.iloc[0]
    assert s["value"] - s["fee"] - s["tax"] == pytest.approx(1000.0, rel=1e-6)
    assert abs(res.totals["books_residual"]) < 1e-6


class BorrowOnce(Policy):
    def __init__(self, day, amount, repay=True):
        self.day, self.amount, self.repay = day, amount, repay

    def setup(self, eng):
        eng.repay_from_contrib = self.repay

    def on_close(self, eng, i):
        if i == self.day:
            eng.borrow(self.amount, i)
        return [MOC(eng.cash)] if eng.cash > 0 else []


def test_borrow_repaid_from_contributions_flows_unchanged():
    px = step_prices(50, 50)
    eng = Engine(px, NO_DIST, px.index[0], px.index[-1], BorrowOnce(5, 1000.0), cost=FREE)
    res = eng.run()
    t = res.totals
    assert [a for _, a in res.flows] == [-500.0] * len(res.flows)          # salary never changes
    assert t["borrowed"] == pytest.approx(1000.0)
    assert t["repaid_from_contrib"] == pytest.approx(1000.0)                # next two quarters repaid it
    assert t["debt_end"] == pytest.approx(0.0)
    assert t["contributed"] - t["repaid_from_contrib"] + t["borrowed"] == pytest.approx(t["spent_on_shares"] + t["cash_end"])
    assert abs(t["books_residual"]) < 1e-6


def test_interest_accrues_and_final_value_is_net_of_debt():
    px = step_prices(50, 50, n=262)
    rate = pd.Series(0.0365, index=px.index)
    eng = Engine(px, NO_DIST, px.index[0], px.index[-1], BorrowOnce(0, 1000.0, repay=False), cost=FREE,
                 debt_rate=rate)
    res = eng.run()
    t = res.totals
    days = (px.index[-1] - px.index[0]).days
    assert t["debt_end"] == pytest.approx(1000 * (1 + 0.0365 * 1 / 365) ** days, rel=2e-3)
    m = summarize(res, FREE)
    assert m["final_value"] == pytest.approx(t["value_end"] - t["debt_end"])
    assert m["interest_paid"] == pytest.approx(t["debt_end"] - 1000.0)


def test_old_runs_unchanged_when_unused():
    px = step_prices()
    from src.benchmarks import BuyHold
    r = run(px, NO_DIST, px.index[0], px.index[-1], BuyHold(), cost=IBKR_TIERED)
    t = r.totals
    assert t["debt_end"] == 0 and t["cgt_paid"] == 0
    assert abs(t["books_residual"]) < 1e-6
    assert t["contributed"] + t["div_gross"] - t["div_tax"] == pytest.approx(t["cash_end"] + t["spent_on_shares"] + t["fees"])
