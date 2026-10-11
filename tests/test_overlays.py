"""
tests/test_overlays.py  -  study 10 overlays do what they say, and the value signal never looks ahead.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmarks import BuyHold                       # noqa: E402
from src.costs import ZERO_FEES, with_fractional         # noqa: E402
from src.engine import Engine                            # noqa: E402
from src.overlays import Overlay, value_signal           # noqa: E402

FREE = with_fractional(ZERO_FEES)
NO_DIST = pd.DataFrame(columns=["ex_date", "pay_date", "amount_eur"])


def flat(n=520):
    idx = pd.bdate_range("2010-01-01", periods=n)
    c = np.full(n, 50.0)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx)


def go(pol, px):
    res = Engine(px, NO_DIST, px.index[0], px.index[-1], pol, cost=FREE).run()
    assert abs(res.totals["books_residual"]) < 1e-6
    return res


def test_trend_filter_skips_then_invests_the_pile():
    px = flat()
    ok = pd.Series(px.index >= px.index[200], index=px.index)          # "downtrend" for the first 200 days
    res = go(Overlay(BuyHold(), "trend_above", trend_ok=ok), px)
    tr = res.trades
    assert (tr["date"] < px.index[200]).sum() == 0
    first = tr.iloc[0]
    assert first["date"] == pd.Timestamp("2011-01-03")                 # next buy day after the trend turned (Oct 2010)
    assert first["value"] == pytest.approx(500 * 5)                     # all 5 waiting quarters invested at once


def test_value_sizing_saves_when_expensive_and_spends_when_cheap():
    px = flat()
    m = pd.Series(np.where(np.arange(len(px)) < 260, 0.5, 1.5), index=px.index)
    res = go(Overlay(BuyHold(), "value", mult=m), px)
    tr = res.trades.set_index("date")["value"]
    assert tr.iloc[0] == pytest.approx(250.0)                           # expensive: half of EUR 500
    assert res.daily["cash"].iloc[250] == pytest.approx(4 * 250.0)      # 4 halves saved after a year
    later = tr[tr.index >= px.index[260]]
    assert later.iloc[0] == pytest.approx(750.0)                        # cheap: 500 + 250 from the pot
    assert res.daily["cash"].iloc[-1] == pytest.approx(0.0, abs=1e-6)   # pot used up in 4 cheap quarters


def test_value_signal_is_point_in_time():
    months = pd.date_range("1980-01-01", "2012-12-01", freq="MS")
    rng = np.random.default_rng(0)
    P = 100 * np.exp(np.cumsum(rng.normal(0.005, 0.04, len(months))))
    sm = pd.DataFrame({"P": P, "E": P / 18, "D": P * 0.02}, index=months)
    days = pd.bdate_range("2005-01-01", "2012-12-31")
    from src.pit import shiller_available
    pit = shiller_available(sm.assign(CAPE=20.0, GS10=4.0)[["E", "D", "CAPE", "GS10"]], days)
    spx = pd.Series(np.interp(np.arange(len(days)), np.linspace(0, len(days), 96), P[-96:]), index=days)
    base = value_signal(spx, sm, pit)
    cut = days[1000]
    sm2 = sm.copy()
    sm2.loc[sm2.index > cut - pd.DateOffset(months=7), ["P", "E", "D"]] *= 3.0     # rewrite the unpublished future
    spx2 = spx.copy()
    spx2[spx2.index >= cut] *= 2.0                                                 # and today's/future prices
    test = value_signal(spx2, sm2, shiller_available(sm2.assign(CAPE=20.0, GS10=4.0)[["E", "D", "CAPE", "GS10"]], days))
    a, b = base.loc[:cut, "e"], test.loc[:cut, "e"]
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)
