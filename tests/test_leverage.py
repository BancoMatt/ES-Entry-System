"""tests/test_leverage.py  -  study 09: target leverage, margin calls, daily-reset fund wipe-out."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.costs import ZERO_FEES, with_fractional        # noqa: E402
from src.engine import Engine                           # noqa: E402
from src.leverage import LeverageTarget, letf_prices, maintenance   # noqa: E402

FREE = with_fractional(ZERO_FEES)
NO_DIST = pd.DataFrame(columns=["ex_date", "pay_date", "amount_eur"])


def prices(path):
    idx = pd.bdate_range("2010-01-01", periods=len(path))
    c = np.asarray(path, float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx)


def go(L, px):
    pol = LeverageTarget(L)
    res = Engine(px, NO_DIST, px.index[0], px.index[-1], pol, cost=FREE).run()
    assert abs(res.totals["books_residual"]) < 1e-6
    return pol, res


def test_maintenance_levels():
    assert maintenance(1) == 0 and maintenance(2) == 0.25 and maintenance(4) == 0.125 and maintenance(10) == 0.05


def test_two_x_after_first_buy():
    pol, res = go(2.0, prices([50.0] * 30))
    d = res.daily.iloc[0]
    assert d["shares"] * 50 == pytest.approx(1000.0)        # EUR 500 own money -> EUR 1,000 position
    assert d["debt"] == pytest.approx(500.0)


def test_one_x_never_borrows():
    pol, res = go(1.0, prices(np.linspace(50, 80, 300)))
    assert res.totals["borrowed"] == 0


def test_margin_call_ends_the_scenario():
    path = [50.0] * 10 + [36.0] * 50                        # -28%: 3x equity/position = (108-66.7... ) below 16.7%
    pol, res = go(3.0, prices(path))
    assert pol.bust == prices(path).index[10]
    assert res.daily["shares"].iloc[-1] == 0


def test_two_x_survives_the_same_drop():
    pol, _ = go(2.0, prices([50.0] * 10 + [36.0] * 50))
    assert pol.bust is None


def test_letf_wiped_by_one_bad_day():
    px = prices([100.0] * 5 + [89.0] * 5)                   # one -11% day
    rate = pd.Series(0.0, index=px.index)
    _, wipe10 = letf_prices(px, NO_DIST, 10, rate, px.index[0], px.index[-1])
    _, wipe5 = letf_prices(px, NO_DIST, 5, rate, px.index[0], px.index[-1])
    assert wipe10 == px.index[5] and wipe5 is None
