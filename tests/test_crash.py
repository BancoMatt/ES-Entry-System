"""
tests/test_crash.py  -  study 11 policies on a hand-made crash:
price 100 for 120 days -> 70 (-30%) for 120 days -> 110 (new high) for the rest.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmarks import BuyHold                                   # noqa: E402
from src.costs import ZERO_FEES, with_fractional                     # noqa: E402
from src.crash import CreditLine, FrontLoad, Panic, Pause, Reserve   # noqa: E402
from src.engine import Engine, run                                   # noqa: E402

FREE = with_fractional(ZERO_FEES)
NO_DIST = pd.DataFrame(columns=["ex_date", "pay_date", "amount_eur"])


def crash_prices():
    idx = pd.bdate_range("2010-01-01", periods=600)
    c = np.r_[np.full(120, 100.0), np.full(120, 70.0), np.full(360, 110.0)]
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx)


def go(pol, **kw):
    px = crash_prices()
    eng = Engine(px, NO_DIST, px.index[0], px.index[-1], pol, cost=FREE, **kw)
    res = eng.run()
    assert abs(res.totals["books_residual"]) < 1e-6
    return res


def test_frontload_fires_once_per_crash_and_is_repaid():
    pol = FrontLoad(0.20, 4, "one")
    res = go(pol)
    assert len(pol.fires) == 1
    d0, dd0, amt = pol.fires[0]
    assert d0 == crash_prices().index[121]                # signal on the first -30% close, buy next close
    assert amt == pytest.approx(2000.0)
    assert res.totals["debt_end"] == pytest.approx(0.0)
    assert res.totals["repaid_from_contrib"] == pytest.approx(2000.0)


def test_frontload_beats_bh_on_a_v_crash():
    a = go(FrontLoad(0.20, 4, "one")).totals["value_end"]
    b = go(BuyHold()).totals["value_end"]
    assert a > b


def test_ladder_fires_each_level_once():
    pol = FrontLoad(0.10, 1, "ladder")
    go(pol)
    assert len(pol.fires) == 1                            # levels -10, -20, -30 all hit on the same day -> one 3-quarter draw
    assert pol.fires[0][2] == pytest.approx(1500.0)


def test_placebo_fires_on_given_days():
    pol = FrontLoad(0.20, 2, fire_days=[10, 300])
    go(pol)
    assert [f[0] for f in pol.fires] == [crash_prices().index[10], crash_prices().index[300]]


def test_pause_skips_buys_in_the_crash_then_invests_the_pile():
    res = go(Pause(0.20, 0.10))
    tr = res.trades
    crash = (tr["date"] > crash_prices().index[120]) & (tr["date"] <= crash_prices().index[239])
    assert crash.sum() == 0
    assert tr[tr["date"] == crash_prices().index[241]]["value"].sum() > 500


def test_panic_sells_everything_and_rebuys():
    pol = Panic(0.20, 0.0)
    res = go(pol)
    s = res.sells
    assert len(s) == 1 and s.iloc[0]["price"] == pytest.approx(70.0)
    assert [e[0] for e in pol.events] == ["sell", "rebuy"]
    assert res.totals["value_end"] < go(BuyHold()).totals["value_end"]


def test_reserve_keeps_cash_then_deploys():
    res = go(Reserve(0.10, 0.20))
    d = res.daily
    assert d["cash"].iloc[50] == pytest.approx(50.0)      # 10% of the first EUR 500 kept aside (March)
    assert d["cash"].iloc[100] == pytest.approx(100.0)    # and of the April one
    assert d["cash"].iloc[125] < 1.0                      # deployed in the crash


def test_credit_line_draws_and_repays_at_new_high():
    px = crash_prices()
    rate = pd.Series(0.05, index=px.index)
    pol = CreditLine(0.20, 0.20, "recovery")
    eng = Engine(px, NO_DIST, px.index[0], px.index[-1], pol, cost=FREE, debt_rate=rate)
    eng.contrib = {eng.dates[0]: 1_000_000.0}
    res = eng.run()
    assert len(pol.draws) == 1
    assert res.totals["debt_end"] == pytest.approx(0.0, abs=1e-6)
    assert res.totals["interest"] > 0
    assert abs(res.totals["books_residual"]) < 1e-3
