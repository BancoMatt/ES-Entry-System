"""
tests/test_strategy.py  -  the dip strategy on the engine.

1. With (practically) unlimited cash, the strategy must buy on EXACTLY the days the
   frozen signal code (src/signals.py) says. Proves the two implementations agree.
2. Books balance and cash never negative, for a spread of configurations.
3. No look-ahead: scrambling the future never changes past trades.
4. E2 buys at the NEXT day's close; deadlines and pieces behave as specified.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.costs import ZERO_FEES, IBKR_TIERED, with_fractional   # noqa: E402
from src.engine import run                                      # noqa: E402
from src.pit import known_at_open_features                      # noqa: E402
from src.signals import Rule, entries, default_grid             # noqa: E402
from src.strategy import DipStrategy                            # noqa: E402
from tests.test_lookahead import fake_prices                    # noqa: E402
from tests.test_signals import scramble                         # noqa: E402

NO_DIST = pd.DataFrame(columns=["ex_date", "pay_date", "amount_eur"])
RICH = 1e8             # "unlimited" yearly budget (EUR 25M per piece)


def _px(n=900, seed=1):
    px = fake_prices(n=n, seed=seed)
    px.index = pd.bdate_range("2008-01-01", periods=len(px))
    return px


@pytest.mark.parametrize("rule", [
    Rule("prev_close", "fixed", 0.0125),
    Rule("high_20d", "fixed", 0.02, "A"),
    Rule("high_10d", "vol", 2.0, "C", cooldown=5),
    Rule("high_20d", "fixed", 0.015, "B"),
])
@pytest.mark.parametrize("style", ["E1", "E2"])
def test_matches_frozen_signal_code_with_unlimited_cash(rule, style):
    px = _px()
    start = px.index[60]
    cm = with_fractional(ZERO_FEES)
    # huge budget split in 100,000 pieces -> cash never runs out, every signal can be bought
    res = run(px, NO_DIST, start, px.index[-1], DipStrategy(rule, style, pieces=100_000, trade_through=0.0),
              cost=cm, per_year=RICH)
    dips = res.trades[res.trades["tag"] == "dip"]
    # both start their rule memory on the same day (features still use the earlier history)
    feat = known_at_open_features(px)
    sig = entries(px.loc[start:], feat.loc[start:], rule, style)
    sig_days = sig.index[sig["n"].fillna(0) > 0]
    if style == "E2":            # E2 buys the NEXT trading day
        pos = [px.index.get_loc(d) + 1 for d in sig_days]
        sig_days = pd.DatetimeIndex([px.index[p] for p in pos if p < len(px)])
    assert list(pd.DatetimeIndex(dips["date"].unique())) == list(sig_days)


@pytest.mark.parametrize("cfg", [
    dict(style="E1", pieces=2, deadline="none"), dict(style="E1", pieces=3, deadline=30),
    dict(style="E2", pieces=1, deadline=60), dict(style="E2", pieces=4, deadline="next"),
])
def test_books_balance_and_cash_never_negative(cfg):
    px = _px()
    dist = pd.DataFrame({"ex_date": [px.index[200], px.index[500]], "pay_date": [px.index[215], px.index[515]],
                         "amount_eur": [0.3, 0.3]})
    for rule in default_grid()[::23]:
        if cfg["style"] == "E2" and rule.ref == "open":
            continue
        res = run(px, dist, px.index[60], px.index[-1], DipStrategy(rule, **cfg), cost=IBKR_TIERED)
        t = res.totals
        assert t["contributed"] + t["div_gross"] - t["div_tax"] == pytest.approx(
            t["cash_end"] + t["spent_on_shares"] + t["fees"], abs=1e-6)
        assert (res.daily["cash"] >= -1e-9).all()


@pytest.mark.parametrize("style", ["E1", "E2"])
def test_future_never_changes_past_trades(style):
    px = _px()
    rule = Rule("high_20d", "fixed", 0.015, "B")
    k = 600
    kw = dict(pieces=2, deadline=60)
    base = run(px, NO_DIST, px.index[60], px.index[-1], DipStrategy(rule, style, **kw)).trades
    p2 = scramble(px, k, include_day_k=False)
    alt = run(p2, NO_DIST, px.index[60], px.index[-1], DipStrategy(rule, style, **kw)).trades
    cut = px.index[k]
    a = base[base["date"] <= cut].reset_index(drop=True)
    b = alt[alt["date"] <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_E2_buys_next_day_close():
    idx = pd.bdate_range("2010-01-04", periods=30)
    c = np.full(30, 100.0)
    c[25] = 97.0                                   # close drops 3% below the 5-day high
    px = pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx)
    cm = with_fractional(ZERO_FEES)
    res = run(px, NO_DIST, idx[0], idx[-1],
              DipStrategy(Rule("high_5d", "fixed", 0.02, "A"), "E2", pieces=1), cost=cm)
    dips = res.trades[res.trades["tag"] == "dip"]
    assert list(dips["date"]) == [idx[26]]
    assert dips["price"].iloc[0] == pytest.approx(100.0)   # day 26 close (spread 0 here)


def test_deadline_invests_waiting_cash():
    px = _px()
    never = Rule("high_20d", "fixed", 0.50, "A")          # a 50% drop never happens
    cm = with_fractional(ZERO_FEES)
    res = run(px, NO_DIST, px.index[60], px.index[-1], DipStrategy(never, "E1", 1, 30), cost=cm)
    dl = res.trades[res.trades["tag"] == "deadline"]
    assert len(dl) > 0 and (res.trades["tag"] == "dip").sum() == 0
    first_contrib = res.daily.index[0]
    assert (dl["date"].iloc[0] - first_contrib).days >= 30
    assert (dl["date"].iloc[0] - first_contrib).days <= 34


def test_pieces_size_orders():
    px = _px()
    cm = with_fractional(ZERO_FEES)
    res = run(px, NO_DIST, px.index[60], px.index[-1],
              DipStrategy(Rule("prev_close", "fixed", 0.005), "E1", pieces=2, trade_through=0.0), cost=cm)
    dips = res.trades[res.trades["tag"] == "dip"]
    assert dips["value"].max() <= 250 + 1e-6
