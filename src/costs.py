"""
src/costs.py  -  what every trade and every euro of income costs you.

One place for all cost rules, so every study uses exactly the same numbers.

  fee per order (IBKR Pro Tiered, Euronext):  min(29, max(1.25, 0.05% x value)) + 0.30
                                               (0.30 = our estimate of exchange/clearing)
  spread: market / market-on-close orders pay half the bid-ask spread (2.5 bps);
          limit orders pay their limit price (or a better open) and no extra spread
  taxes:  26% on each dividend when paid; 26% on the gain if the position is sold
"""
import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CostModel:
    name: str = "ibkr_tiered"
    fee_pct: float = 0.0005
    fee_min: float = 1.25
    fee_max: float = 29.0          # use math.inf for "no maximum"
    fee_extra: float = 0.30        # exchange + clearing estimate, per order
    half_spread: float = 0.00025   # 2.5 bps, market / MOC orders only
    div_tax: float = 0.26
    cgt: float = 0.26              # capital gains tax when selling (used for "after selling" figures)
    fractional: bool = False       # baseline: whole shares only

    def fee(self, value: float) -> float:
        if value <= 0:
            return 0.0
        return min(self.fee_max, max(self.fee_min, self.fee_pct * value)) + self.fee_extra

    def affordable(self, budget: float, price: float) -> float:
        """How many shares a budget buys at `price`, AFTER paying the fee out of the same budget."""
        if budget <= 0 or price <= 0:
            return 0.0
        if self.fractional:
            v = budget
            for _ in range(8):                   # value + fee(value) = budget  (converges fast)
                v = budget - self.fee(v)
            return max(v, 0.0) / price
        n = math.floor(budget / price)
        while n > 0 and n * price + self.fee(n * price) > budget + 1e-9:
            n -= 1
        return float(n)


IBKR_TIERED = CostModel()
IBKR_FIXED = CostModel(name="ibkr_fixed", fee_min=3.0, fee_max=math.inf, fee_extra=0.0)
ZERO_FEES = CostModel(name="zero_fees", fee_pct=0.0, fee_min=0.0, fee_extra=0.0, half_spread=0.0)


def with_fractional(cm: CostModel) -> CostModel:
    return replace(cm, name=cm.name + "_fractional", fractional=True)
