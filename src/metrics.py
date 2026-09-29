"""
src/metrics.py  -  the scorecard for one run.

XIRR (money-weighted return) is the headline, because money arrives over time:
it's the single yearly rate that makes all contributions grow into the final value.
  "before selling": final value = shares x last close + cash
  "after selling":  minus 26% tax on the gain (value - cost basis), if positive

Max drawdown uses a TIME-WEIGHTED index (strips out new contributions), otherwise
every new EUR 500 would look like a gain and hide real drops.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq


def xirr(flows):
    """flows: list of (date, amount); money in negative, money out positive."""
    if not flows:
        return np.nan
    t0 = min(d for d, _ in flows)
    years = np.array([(d - t0) / pd.Timedelta(days=365.25) for d, _ in flows])
    amts = np.array([a for _, a in flows], float)

    def npv(r):
        return np.sum(amts / (1 + r) ** years)
    try:
        return brentq(npv, -0.99, 10.0)
    except ValueError:
        return np.nan


def summarize(res, cost) -> dict:
    tot, daily = res.totals, res.daily
    end, v_end = tot["end_date"], tot["value_end"]
    gain = v_end - tot["cash_end"] - tot["cost_basis"]
    v_after = v_end - max(gain, 0.0) * cost.cgt
    ext = daily["contributed"].diff().fillna(daily["contributed"])
    prev = daily["value"].shift(1)
    r = ((daily["value"] - ext) / prev - 1).where(prev > 0).fillna(0.0)
    twr = (1 + r).cumprod()
    dd = (twr / twr.cummax() - 1).min()
    tr = res.trades
    return {
        "policy": res.name,
        "start": daily.index[0].date(), "end": end.date(),
        "xirr_before_sell": xirr(res.flows + [(end, v_end)]),
        "xirr_after_sell": xirr(res.flows + [(end, v_after)]),
        "final_value": v_end,
        "contributed": tot["contributed"],
        "profit": v_end - tot["contributed"],
        "fees": tot["fees"],
        "fees_pct_of_contrib": tot["fees"] / tot["contributed"],
        "div_tax": tot["div_tax"],
        "avg_cash_pct": float((daily["cash"] / daily["value"]).where(daily["value"] > 0).mean()),
        "max_drawdown": float(dd),
        "n_buys": len(tr),
        "avg_order_eur": float(tr["value"].mean()) if len(tr) else 0.0,
        "avg_price_paid": float(tr["value"].sum() / tr["shares"].sum()) if len(tr) else np.nan,
        "cost_model": cost.name,
    }
