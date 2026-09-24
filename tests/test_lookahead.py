"""
tests/test_lookahead.py  -  automatic proof that no feature peeks into the future.

Run from project root:   python -m pytest -q

Uses a random fake price history, so it runs anywhere in a second, no data needed.
Every future feature/signal function added to the project gets a test here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pit import known_at_open_features, lookahead_check, shiller_available  # noqa: E402


def fake_prices(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.004, n))),
        "low": np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.004, n))),
        "close": close,
        "usd_per_eur": 1.1 + rng.normal(0, 0.01, n),
        "tnx": 3 + rng.normal(0, 0.1, n),
    }, index=idx)


def test_features_never_use_today_close_or_future():
    px = fake_prices()
    fails = lookahead_check(known_at_open_features, px, cutoffs=[250, 600, 1000, 1400])
    assert fails == [], f"look-ahead detected: {fails}"


def test_the_check_itself_catches_a_cheater():
    """Sanity: a feature that uses today's close MUST be caught."""
    def cheating(px):
        f = known_at_open_features(px)
        f["cheat"] = px["close"]            # today's close is not known at the open
        return f
    fails = lookahead_check(cheating, fake_prices(), cutoffs=[500])
    assert any(col == "cheat" for _, col, _ in fails)


def test_shiller_lag():
    months = pd.date_range("2000-01-01", "2010-12-01", freq="MS")
    sh = pd.DataFrame({"E": np.arange(len(months), dtype=float)}, index=months)
    days = pd.bdate_range("2001-01-01", "2010-12-31")
    av = shiller_available(sh, days, lag_months=6)
    # month m is usable only from the 1st of m + 7 months
    gap = (av.index.to_period("M") - av["shiller_month"].dt.to_period("M")).map(lambda x: x.n)
    assert gap.min() >= 7
