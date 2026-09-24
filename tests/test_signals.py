"""
tests/test_signals.py  -  entry rules must not peek into the future.

Two properties, for every rule in the grid:
 1. The LIMIT PRICE for day t is fixed at the open: scrambling day t's close/high/low
    (and everything after) must not change any limit on or before day t.
 2. Whether a buy happened on day <= t must not change when days AFTER t are scrambled.
Plus hand-made examples that check the A/B/C logic does what the docstring says.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pit import known_at_open_features          # noqa: E402
from src.signals import Rule, entries, default_grid  # noqa: E402
from tests.test_lookahead import fake_prices         # noqa: E402


def scramble(px, k, include_day_k):
    p = px.copy()
    rng = np.random.default_rng(7)
    start = k if include_day_k else k + 1
    for col in ["high", "low", "close"]:
        v = p[col].to_numpy(float, copy=True)
        v[start:] *= rng.uniform(0.8, 1.2, len(v) - start)
        p[col] = v
    o = p["open"].to_numpy(float, copy=True)
    o[k + 1:] *= rng.uniform(0.8, 1.2, len(o) - k - 1)
    p["open"] = o
    p["low"] = p[["low", "open", "close"]].min(axis=1)
    p["high"] = p[["high", "open", "close"]].max(axis=1)
    return p


def test_limits_and_fills_have_no_lookahead():
    px = fake_prices(n=900)
    px.index = pd.bdate_range("2008-01-01", periods=len(px))   # after OPEN_RELIABLE_FROM
    base_f = known_at_open_features(px)
    for r in default_grid()[::7]:                                # a spread of rules
        base = entries(px, base_f, r)
        for k in (300, 600):
            p1 = scramble(px, k, include_day_k=True)
            e1 = entries(p1, known_at_open_features(p1), r)
            assert np.allclose(base["limit"].iloc[:k + 1], e1["limit"].iloc[:k + 1], equal_nan=True), r.name
            p2 = scramble(px, k, include_day_k=False)
            e2 = entries(p2, known_at_open_features(p2), r)
            assert np.allclose(base["n"].iloc[:k + 1], e2["n"].iloc[:k + 1], equal_nan=True), r.name


def _toy(closes, lows=None):
    idx = pd.bdate_range("2010-01-04", periods=len(closes))
    c = np.array(closes, float)
    lo = np.array(lows if lows is not None else closes, float)
    px = pd.DataFrame({"open": c, "high": c, "low": lo, "close": c}, index=idx)
    return px, known_at_open_features(px)


def test_rule_A_one_entry_until_new_high():
    # high 100, drops to 96 twice without a new high -> only ONE entry; new high then drop -> second
    closes = [100] * 5 + [96, 99, 96, 99, 101, 101, 97]
    px, f = _toy(closes)
    e = entries(px, f, Rule("high_5d", "fixed", 0.03, "A"))
    assert e["n"].fillna(0).sum() == 2


def test_rule_B_ladder_levels():
    # drop -2%, -4%, -6% from 100 with 2% steps -> 3 entries, one per level
    closes = [100] * 5 + [98, 96, 94, 94]
    px, f = _toy(closes)
    e = entries(px, f, Rule("high_5d", "fixed", 0.02, "B"))
    assert e["n"].fillna(0).sum() == 3


def test_gap_down_fills_at_open():
    closes = [100] * 5 + [95]
    px, f = _toy(closes)                         # opens at 95, limit at 98.75
    e = entries(px, f, Rule("prev_close", "fixed", 0.0125))
    assert e["fill"].iloc[-1] == 95
