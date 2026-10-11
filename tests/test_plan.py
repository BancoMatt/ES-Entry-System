"""tests/test_plan.py  -  study 13: plan schedule, loan re-set, ETF split without selling, margin call, taxes."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.plan as plan                                      # noqa: E402
from src.plan import const, drawdown_stats, glide, plan_matt, simulate   # noqa: E402


@pytest.fixture(autouse=True)
def no_fees(monkeypatch):
    monkeypatch.setattr(plan, "FEE", 0.0)


def arrays(r, T, contrib_every=1, c=100.0, rate=0.0, r2=None):
    r1 = np.full((1, T), r, float)
    r1[0, 0] = 0.0
    r2 = 2 * r1 if r2 is None else r2
    C = np.zeros((1, T))
    C[0, ::contrib_every] = c
    return r1, r2, np.full((1, T), rate), C


def test_plan_schedule():
    assert plan_matt([0, 3]).tolist() == [500, 500]
    assert plan_matt([4, 15]).tolist() == [1200, 1200]
    assert plan_matt([16, 19, 20]).tolist() == [3000, 3000, 3150]
    assert glide([0, 19, 20]).tolist() == [1.5, 1.5, 1.25]


def test_1x_is_plain_buy_and_hold():
    r1, r2, mr, C = arrays(0.01, 4)
    res = simulate(r1, r2, mr, C, np.ones((1, 4)), "margin")
    expect = 100 * (1.01 ** 3 + 1.01 ** 2 + 1.01 + 1)
    assert res["final_before"][0] == pytest.approx(expect)
    assert res["interest"][0] == 0


def test_loan_resets_to_target_on_contribution_days():
    r1, r2, mr, C = arrays(0.05, 6)
    res = simulate(r1, r2, mr, C, np.full((1, 6), 1.25), "margin")
    assert np.allclose(res["exposure"][0], 1.25, atol=1e-9)


def test_loan_sells_after_a_loss_and_taxes_only_gains():
    r1 = np.array([[0.0, 0.5, -0.3]])
    C = np.array([[100.0, 0.0, 1e-9]])                         # tiny contribution = re-set day
    res = simulate(r1, 2 * r1, np.zeros((1, 3)), C, np.full((1, 3), 1.5), "margin")
    # bought 150 (debt 50) -> 225 -> 157.5; equity 107.5 -> target 161.25 > 157.5: buys, no sale
    assert res["tax_on_rebalancing"][0] == 0
    r1 = np.array([[0.0, 0.0, -0.3]])
    res = simulate(r1, 2 * r1, np.zeros((1, 3)), C, np.full((1, 3), 1.5), "margin")
    assert res["tax_on_rebalancing"][0] == 0                   # sold at a loss: no tax
    assert res["exposure"][0, -1] == pytest.approx(1.5, abs=1e-6)


def test_margin_call_ends_at_zero():
    r1 = np.array([[0.0, -0.40]])
    C = np.array([[100.0, 0.0]])
    res = simulate(r1, 2 * r1, np.zeros((1, 2)), C, np.full((1, 2), 1.5), "margin")
    # 150 position, 50 debt -> 90 position, equity 40 = 44% > 25%: survives
    assert res["alive"][0]
    r1 = np.array([[0.0, -0.60]])
    res = simulate(r1, 2 * r1, np.zeros((1, 2)), C, np.full((1, 2), 1.5), "margin")
    # 60 position, 10 equity = 17% < 25%: called
    assert not res["alive"][0] and res["final_after"][0] == 0 and res["residual"][0] == pytest.approx(10)


def test_etf_split_never_sells_and_hits_target_with_new_money():
    r1, r2, mr, C = arrays(0.0, 3)
    res = simulate(r1, r2, mr, C, np.full((1, 3), 1.25), "etf")
    assert np.allclose(res["exposure"][0], 1.25)
    # 2x fund rallies: exposure drifts up, new money goes to the 1x fund only, nothing sold
    r1 = np.array([[0.0, 0.2, 0.0]])
    res = simulate(r1, 2 * r1, np.zeros((1, 3)), np.array([[100.0, 0.0, 10.0]]), np.full((1, 3), 1.25), "etf")
    assert res["tax_on_rebalancing"][0] == 0
    assert res["exposure"][0, -1] > 1.25


def test_end_tax_per_fund_no_offset():
    r1 = np.array([[0.0, 0.10]])
    r2 = np.array([[0.0, -0.20]])
    res = simulate(r1, r2, np.zeros((1, 2)), np.array([[100.0, 0.0]]), np.full((1, 2), 1.5), "etf")
    # 50 -> 55 (gain 5), 50 -> 40 (loss 10): tax on the 5 only
    assert res["end_tax"][0] == pytest.approx(0.26 * 5)


def test_interest_accrues_on_the_loan():
    r1, r2, mr, C = arrays(0.0, 3, contrib_every=10, rate=0.01)
    res = simulate(r1, r2, mr, C, np.full((1, 3), 2.0), "margin")
    assert res["interest"][0] == pytest.approx(100 * 0.01 + 101 * 0.01)


def test_drawdown_counts_market_not_contributions():
    pre = np.array([[100.0, 90.0, 220.0]])
    post = np.array([[100.0, 190.0, 200.0]])                    # +100 contribution at step 1
    mdd, under = drawdown_stats(pre, post, 1)
    assert mdd[0] == pytest.approx(-0.10)
    assert under[0] == 1
