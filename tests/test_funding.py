"""
tests/test_funding.py  -  the study 06 / calendar policies do exactly what they claim.
Flat prices + zero fees + fractional shares, so every number can be checked by hand.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmarks import (BuyHold, InvestOnDays, SpreadBuys,          # noqa: E402
                            extreme_day_of_year, nth_day_of_year)
from src.costs import ZERO_FEES, with_fractional                         # noqa: E402
from src.engine import Engine, run                                       # noqa: E402

FREE = with_fractional(ZERO_FEES)
NO_DIST = pd.DataFrame(columns=["ex_date", "pay_date", "amount_eur"])


def prices(start="2010-01-01", end="2012-12-31"):
    idx = pd.bdate_range(start, end)
    c = np.full(len(idx), 50.0)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx)


def test_nth_day_of_year():
    px = prices()
    first = nth_day_of_year(px.index, 1)
    assert [px.index[i] for i in first] == [pd.Timestamp("2010-01-01"), pd.Timestamp("2011-01-03"),
                                            pd.Timestamp("2012-01-02")]
    third = nth_day_of_year(px.index, 3)
    assert all(t - f == 2 for f, t in zip(first, third))


def test_extreme_day_is_lowest_and_highest_close():
    px = prices()
    px.loc["2011-06-15", "close"] = 40.0
    px.loc["2011-09-15", "close"] = 60.0
    lo = extreme_day_of_year(px.index, px["close"].to_numpy(), "min")
    hi = extreme_day_of_year(px.index, px["close"].to_numpy(), "max")
    assert px.index[lo[1]] == pd.Timestamp("2011-06-15")
    assert px.index[hi[1]] == pd.Timestamp("2011-09-15")


def test_annual_lump_sum_buys_once_a_year():
    px = prices()
    r = run(px, NO_DIST, px.index[0], px.index[-1], BuyHold(), cost=FREE, freq="A")
    assert len(r.trades) == 3
    assert r.trades["value"].tolist() == pytest.approx([2000.0] * 3)


def test_spread_over_quarters_with_yearly_money():
    px = prices()
    r = run(px, NO_DIST, px.index[0], px.index[-1], SpreadBuys("Q"), cost=FREE, freq="A")
    assert len(r.trades) == 12
    assert r.trades["value"].tolist() == pytest.approx([500.0] * 12)
    assert r.totals["cash_end"] == pytest.approx(0.0, abs=1e-9)


def test_save_up_waits_and_invests_in_january():
    px = prices()
    jan = lambda eng: nth_day_of_year(eng.dates, 1)                      # noqa: E731
    r = run(px, NO_DIST, px.index[0], px.index[-1], InvestOnDays(jan), cost=FREE, freq="Q")
    # Jan 2010: 500 | Jan 2011: 1,500 saved + 500 | Jan 2012: same | last 1,500 never invested
    assert r.trades["value"].tolist() == pytest.approx([500.0, 2000.0, 2000.0])
    assert r.totals["cash_end"] == pytest.approx(1500.0)
    assert r.totals["avg_wait_days"] > 90


def test_custom_arrival_days_override_schedule():
    px = prices()
    days = nth_day_of_year(px.index, 10)
    eng = Engine(px, NO_DIST, px.index[0], px.index[-1], BuyHold(), cost=FREE)
    eng.contrib = {eng.dates[i]: 2000.0 for i in days}
    r = eng.run()
    assert [d for d, _ in r.flows] == [px.index[i] for i in days]
    assert r.trades["date"].tolist() == [px.index[i] for i in days]
    assert r.totals["avg_wait_days"] == pytest.approx(0.0)
