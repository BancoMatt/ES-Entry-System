"""
tests/test_engine.py  -  the accountant must never be wrong.

Uses small made-up price histories, so every answer can be worked out by hand.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmarks import BuyHold, MonthlyDCA              # noqa: E402
from src.costs import IBKR_TIERED, IBKR_FIXED, ZERO_FEES, with_fractional  # noqa: E402
from src.engine import Engine, Limit, Policy, run          # noqa: E402
from src.metrics import summarize, xirr                    # noqa: E402
from src import data                                       # noqa: E402


def flat_prices(start="2010-01-01", end="2012-12-31", price=50.0, drift=0.0):
    idx = pd.bdate_range(start, end)
    c = price * np.exp(drift * np.arange(len(idx)))
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c}, index=idx)


NO_DIST = pd.DataFrame(columns=["ex_date", "pay_date", "amount_eur"])


# ---------------- costs
def test_fee_formula():
    assert IBKR_TIERED.fee(67) == pytest.approx(1.55)            # minimum applies
    assert IBKR_TIERED.fee(10_000) == pytest.approx(5.30)        # 0.05% + 0.30
    assert IBKR_TIERED.fee(100_000) == pytest.approx(29.30)      # capped at 29
    assert IBKR_FIXED.fee(67) == pytest.approx(3.00)
    assert ZERO_FEES.fee(67) == 0


def test_whole_shares_never_overspend():
    for budget in [60, 100, 166.67, 500, 2000]:
        n = IBKR_TIERED.affordable(budget, 67.0)
        assert n == int(n)
        assert n * 67 + (IBKR_TIERED.fee(n * 67) if n else 0) <= budget + 1e-9
        # one more share would NOT fit
        assert (n + 1) * 67 + IBKR_TIERED.fee((n + 1) * 67) > budget


def test_fractional_spends_whole_budget():
    cm = with_fractional(IBKR_TIERED)
    n = cm.affordable(500, 67.0)
    assert n * 67 + cm.fee(n * 67) == pytest.approx(500, abs=1e-6)


# ---------------- xirr
def test_xirr_simple():
    d0 = pd.Timestamp("2020-01-01")
    assert xirr([(d0, -100), (d0 + pd.Timedelta(days=365.25), 110)]) == pytest.approx(0.10, abs=1e-6)


# ---------------- engine accounting
def _check_books(res):
    t = res.totals
    left = t["contributed"] + t["div_gross"] - t["div_tax"]
    right = t["cash_end"] + t["spent_on_shares"] + t["fees"]
    assert left == pytest.approx(right, abs=1e-6)
    assert (res.daily["cash"] >= -1e-9).all()


@pytest.mark.parametrize("policy", [BuyHold(), MonthlyDCA(1), MonthlyDCA(15), MonthlyDCA(28)])
def test_books_balance_and_cash_never_negative(policy):
    px = flat_prices(drift=0.0003)
    dist = pd.DataFrame({"ex_date": pd.to_datetime(["2010-06-15", "2011-06-15"]),
                         "pay_date": pd.to_datetime(["2010-06-30", "2011-06-30"]),
                         "amount_eur": [0.2, 0.2]})
    res = run(px, dist, px.index[0], px.index[-1], policy)
    _check_books(res)


def test_contributions_quarterly_500():
    px = flat_prices()
    res = run(px, NO_DIST, px.index[0], px.index[-1], BuyHold())
    assert res.totals["contributed"] == pytest.approx(500 * 12)       # 3 years x 4


def test_buy_hold_at_constant_price_zero_fee_returns_zero():
    px = flat_prices()
    cm = with_fractional(ZERO_FEES)
    res = run(px, NO_DIST, px.index[0], px.index[-1], BuyHold(), cost=cm)
    assert summarize(res, cm)["xirr_before_sell"] == pytest.approx(0.0, abs=1e-9)


def test_dca_splits_quarter_in_three():
    px = flat_prices(price=10.0)
    cm = with_fractional(ZERO_FEES)
    res = run(px, NO_DIST, px.index[0], px.index[-1], MonthlyDCA(1), cost=cm)
    first_q = res.trades[res.trades["date"] < "2010-04-01"]
    assert len(first_q) == 3
    assert first_q["value"].sum() == pytest.approx(500, abs=1e-6)
    assert first_q["value"].iloc[0] == pytest.approx(500 / 3, abs=1e-6)


def test_dividend_paid_on_pay_date_to_shares_held_on_ex_date():
    px = flat_prices(price=10.0)
    cm = with_fractional(ZERO_FEES)
    dist = pd.DataFrame({"ex_date": [pd.Timestamp("2010-02-15")],
                         "pay_date": [pd.Timestamp("2010-03-01")], "amount_eur": [1.0]})
    res = run(px, dist, px.index[0], px.index[-1], BuyHold(), cost=cm)
    d = res.daily
    # 50 shares bought on 2010-01-01 -> gross 50, net 37 (26% tax) arrives 1 March, not before
    assert d.loc["2010-02-26", "cash"] == pytest.approx(0, abs=1e-9)
    assert d.loc["2010-03-01", "cash"] == pytest.approx(37.0)
    assert res.totals["div_tax"] == pytest.approx(13.0)


def test_limit_order_trade_through_and_gap_fill():
    px = flat_prices(price=100.0)
    px.loc[px.index[5], ["open", "high", "low", "close"]] = [100.0, 100.0, 98.0, 99.0]   # low 98
    px.loc[px.index[8], ["open", "high", "low", "close"]] = [95.0, 96.0, 94.0, 95.0]     # gap down

    class P(Policy):
        name = "p"

        def __init__(self, tt):
            self.tt = tt

        def on_open(self, eng, i):
            return [Limit(98.0, 200, trade_through=self.tt)] if i in (5, 8) else []

    cm = with_fractional(ZERO_FEES)
    touch = run(px, NO_DIST, px.index[0], px.index[-1], P(0.0), cost=cm).trades
    strict = run(px, NO_DIST, px.index[0], px.index[-1], P(0.0005), cost=cm).trades
    assert list(touch["price"]) == [98.0, 95.0]      # day 5 at the limit, day 8 at the (better) open
    assert list(strict["price"]) == [95.0]           # low 98 == limit: not 0.05% through -> no fill


# ---------------- holdout lock
def test_holdout_is_locked():
    with pytest.raises(PermissionError):
        data.load("holdout")
