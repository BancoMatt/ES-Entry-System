"""
src/leverage.py  -  study 09: what leverage does to EUR 500 a quarter.

TWO ways to be leveraged:

1. MARGIN LOAN with a target leverage L (LeverageTarget)
   On every contribution day the account is set back to L x your net worth: borrow and buy
   after gains, sell (26% tax on any gain) and repay after losses. Idle cash repays the loan.
   Interest = ECB deposit rate + 1.5% (engine debt rate).
   MARGIN CALL when equity / position falls below min(25%, half the starting margin):
     1.2-2x: 25% (standard margin account) | 3x 16.7% | 4x 12.5% | 5x 10% | 10x 5%
   Study rule (Matt): a margin call ENDS the scenario at zero. Whatever the forced sale leaves
   is reported, but counted as lost.

2. DAILY-RESET LEVERAGED ETF (pre-registered model, §11)  letf_prices()
   daily return = L x r - (L-1) x (ECB rate + 0.5%)/252 - 0.6%/252
   (r = IUSA total return that day). No margin calls, but a single day with L x r <= -100%
   wipes the fund out, and choppy markets eat it slowly (volatility decay).
"""
import numpy as np
import pandas as pd

from src.engine import MOC, Policy, Sell


def maintenance(L):
    return 0.0 if L <= 1 else min(0.25, 0.5 / L)


class LeverageTarget(Policy):
    def __init__(self, L):
        self.L = float(L)
        self.mm = maintenance(self.L)
        self.name = f"margin_{self.L:g}x"

    def setup(self, eng):
        self.contrib_idx = {eng.dates.get_loc(d) for d in eng.contrib}
        self.bust, self.residual = None, None
        self.max_lev = 0.0
        self.lev = []

    def on_close(self, eng, i):
        if self.bust is not None:
            return []
        pos = eng.shares * eng.c[i]
        eq = pos + eng.cash - eng.debt
        if self.L > 1 and pos > 0 and eng.debt > 0 and eq / pos < self.mm:
            self.bust = eng.dates[i]
            return [Sell(1.0, tag="margin_call")]
        if i not in self.contrib_idx or eq <= 0:
            return []
        if self.L <= 1:
            return [MOC(eng.cash, "bh")] if eng.cash > 0 else []      # 1x = plain quarterly B&H
        need = self.L * eq - pos
        if need > 0:
            if need > eng.cash + 1e-6:
                eng.borrow(need - eng.cash, i)
            return [MOC(min(need, eng.cash), "lever")]
        if need < 0 and pos > 0:
            return [Sell(fraction=min(1.0, -need / pos), tag="delever")]
        return []

    def after_close(self, eng, i, filled):
        if eng.debt > 0 and eng.cash > 0:
            eng.repay(eng.cash, i)
        if self.bust is not None and self.residual is None:
            self.residual = eng.cash - eng.debt
        pos = eng.shares * eng.c[i]
        eq = pos + eng.cash - eng.debt
        lev = pos / eq if eq > 0 and self.bust is None else np.nan
        self.lev.append(lev)
        if lev == lev:
            self.max_lev = max(self.max_lev, lev)


def letf_prices(px, dist, L, rate, start, end):
    """Daily-reset L x fund on IUSA total return. Returns (price series from 100, wipe date or None)."""
    c = px["close"]
    d_amt = (dist.set_index(pd.to_datetime(dist["ex_date"]))["amount_eur"].groupby(level=0).sum()
             .reindex(c.index).fillna(0.0))
    r = ((c + d_amt) / c.shift(1) - 1).loc[start:end].fillna(0.0)
    rr = rate.sort_index().reindex(r.index, method="ffill").fillna(0.0)
    R = L * r - (L - 1) * (rr + 0.005) / 252 - 0.006 / 252
    R.iloc[0] = 0.0
    dead = R <= -1
    wipe = R.index[dead.argmax()] if dead.any() else None
    price = 100 * (1 + R.clip(lower=-1)).cumprod()
    return price, wipe
