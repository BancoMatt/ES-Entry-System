"""
src/crash.py  -  study 11, the crash playbook: what to do when the market falls.

Every policy keeps quarterly B&H (invest each EUR 500 on arrival) and adds ONE behaviour.
"Drawdown" = how far today's close is below the highest close so far (all history, so a
period starting in a crash knows it's in a crash). Decided on the close of day t, acted on
at the close of day t+1, like E2: no look-ahead.

  Pause       stop buying while drawdown <= -20%; restart (investing the pile) when the market
              recovers to within 10% of its high, or to a new high.
  Panic       sell everything at -20% (26% tax on the gain); buy back with everything when the
              market recovers to within 10% of its high, or to a new high.
  Reserve     keep 10% of every contribution aside in cash; invest the whole reserve at -20%;
              re-arm at a new high. (Control line: a reserve is a delayed lump sum.)
  FrontLoad   at -X%, invest the next N quarters' money now, borrowed from the emergency fund;
              the next contributions repay it. One-shot per crash, or a ladder (another N at
              every further -10%). At most `cap` quarters outstanding. Forgone interest on the
              emergency fund = the engine's debt rate (0% plain cash, ECB rate in a deposit).
              With `fire_days` it fires on those days instead (random-date PLACEBO).
  CreditLine  a big investor (lump sum, no salary). At -X%, borrow f x the portfolio and buy,
              paying ECB rate + 1.5%. Repay at the next all-time high by selling (taxed), or
              carry the loan to the end. If the loan ever exceeds 50% of the portfolio the
              lender forces a sale back to 40% (margin call).
"""
import numpy as np
import pandas as pd

from src.engine import MOC, Policy, Sell


def drawdown(eng):
    """Close vs highest close so far (incl. history before the period), for each day of the period."""
    c = eng.px_all["close"]
    dd = c / c.cummax() - 1
    return dd.reindex(eng.dates).to_numpy(float)


class _Base(Policy):
    """Quarterly B&H core: on contribution days invest whatever cash is free."""
    def setup(self, eng):
        self.contrib_idx = {eng.dates.get_loc(d) for d in eng.contrib}
        self.dd = drawdown(eng)

    def free_cash(self, eng):
        return eng.cash

    def core(self, eng, i):
        if i in self.contrib_idx and self.free_cash(eng) > 0:
            return [MOC(self.free_cash(eng), "bh")]
        return []


class Pause(_Base):
    def __init__(self, enter=0.20, resume=0.10):
        self.enter, self.resume = enter, resume
        self.name = f"pause_-{enter:.0%}_until_" + ("new_high" if resume == 0 else f"-{resume:.0%}")

    def setup(self, eng):
        super().setup(eng)
        self.paused, self.go = False, False

    def on_close(self, eng, i):
        if self.go:
            self.go = False
            return [MOC(eng.cash, "resume")]
        return [] if self.paused else self.core(eng, i)

    def after_close(self, eng, i, filled):
        if not self.paused and self.dd[i] <= -self.enter:
            self.paused = True
        elif self.paused and self.dd[i] >= -self.resume - 1e-12:
            self.paused, self.go = False, True


class Panic(_Base):
    def __init__(self, enter=0.20, rebuy=0.10):
        self.enter, self.rebuy = enter, rebuy
        self.name = f"panic_sell_-{enter:.0%}_rebuy_" + ("new_high" if rebuy == 0 else f"-{rebuy:.0%}")

    def setup(self, eng):
        super().setup(eng)
        self.out, self.sell, self.buy = False, False, False
        self.events = []

    def on_close(self, eng, i):
        if self.sell:
            self.sell = False
            self.events.append(("sell", eng.dates[i], self.dd[i]))
            return [Sell(1.0, tag="panic")]
        if self.buy:
            self.buy = False
            self.events.append(("rebuy", eng.dates[i], self.dd[i]))
            return [MOC(eng.cash, "rebuy")]
        return [] if self.out else self.core(eng, i)

    def after_close(self, eng, i, filled):
        if not self.out and self.dd[i] <= -self.enter:
            self.out, self.sell = True, True
        elif self.out and not self.sell and self.dd[i] >= -self.rebuy - 1e-12:
            self.out, self.buy = False, True


class Reserve(_Base):
    def __init__(self, share=0.10, trigger=0.20):
        self.share, self.trigger = share, trigger
        self.name = f"reserve_{share:.0%}_deploy_-{trigger:.0%}"

    def setup(self, eng):
        super().setup(eng)
        self.reserve, self.armed, self.go = 0.0, True, False
        self.per_contrib = eng.per_year / {"A": 1, "Q": 4, "M": 12}[eng.freq]

    def free_cash(self, eng):
        return max(eng.cash - self.reserve, 0.0)

    def on_close(self, eng, i):
        if i in self.contrib_idx:
            self.reserve += self.share * self.per_contrib
        if self.go:
            self.go, self.reserve = False, 0.0
            return [MOC(eng.cash, "deploy")]
        return self.core(eng, i)

    def after_close(self, eng, i, filled):
        self.reserve = min(self.reserve, eng.cash)
        if self.armed and self.dd[i] <= -self.trigger:
            self.armed, self.go = False, True
        elif not self.armed and self.dd[i] >= -1e-12:
            self.armed = True


class FrontLoad(_Base):
    def __init__(self, trigger=0.20, quarters=4, mode="one", cap=8, step=0.10, fire_days=None):
        self.trigger, self.quarters, self.mode, self.cap, self.step = trigger, quarters, mode, cap, step
        self.fire_days = None if fire_days is None else set(int(x) for x in fire_days)
        tag = "placebo" if fire_days is not None else mode
        self.name = f"frontload_-{trigger:.0%}_N{quarters}_{tag}"

    def setup(self, eng):
        super().setup(eng)
        eng.repay_from_contrib = True
        self.q = eng.per_year / 4
        lv = [self.trigger] if self.mode == "one" else list(np.arange(self.trigger, 0.95, self.step))
        self.levels = [round(x, 6) for x in lv]
        self.fired = set()
        self.pending = 0
        self.fires = []                       # (date, drawdown, euros borrowed)

    def _borrow_and_buy(self, eng, i, n_quarters):
        room = self.cap * self.q - eng.debt
        amt = min(n_quarters * self.q, room)
        if amt < 0.5 * self.q:
            return []
        eng.borrow(amt, i)
        self.fires.append((eng.dates[i], self.dd[i], amt))
        return [MOC(eng.cash, "frontload")]

    def on_close(self, eng, i):
        if self.fire_days is not None:
            if i in self.fire_days:
                return self._borrow_and_buy(eng, i, self.quarters)
            return self.core(eng, i)
        if self.pending:
            n, self.pending = self.pending, 0
            out = self._borrow_and_buy(eng, i, n)
            if out:
                return out
        return self.core(eng, i)

    def after_close(self, eng, i, filled):
        if self.fire_days is not None:
            return
        if self.dd[i] >= -1e-12:
            self.fired.clear()                # new high: re-arm every level
            return
        new = [lv for lv in self.levels if self.dd[i] <= -lv and lv not in self.fired]
        if new and i + 1 < len(eng.dates):
            self.fired.update(new)
            self.pending = self.quarters * len(new)


class CreditLine(Policy):
    def __init__(self, trigger=0.20, frac=0.20, repay="recovery", max_ltv=0.50, reset_ltv=0.40):
        self.trigger, self.frac, self.repay_mode = trigger, frac, repay
        self.max_ltv, self.reset_ltv = max_ltv, reset_ltv
        self.name = f"credit_-{trigger:.0%}_borrow{frac:.0%}_{repay}"

    def setup(self, eng):
        self.dd = drawdown(eng)
        self.armed, self.go, self.payback = True, False, False
        self.draws, self.margin_calls = [], []
        self.ltv = []

    def on_close(self, eng, i):
        out = []
        gross = eng.shares * eng.c[i] + eng.cash
        if eng.debt > self.max_ltv * gross > 0:                          # margin call, at today's close
            need = (eng.debt - self.reset_ltv * gross) / (1 - self.reset_ltv)
            self.margin_calls.append((eng.dates[i], eng.debt / gross, need))
            out.append(Sell(amount=need, tag="margin_call"))
            self.payback = True
        if self.payback and eng.debt > 0 and not out:
            out.append(Sell(amount=eng.debt, tag="repay"))
        if self.go:
            self.go = False
            net = gross - eng.debt
            amt = self.frac * net
            if amt > 0 and (eng.debt + amt) / (gross + amt) <= 0.30 + 1e-9:   # never draw above 30% LTV
                eng.borrow(amt, i)
                self.draws.append((eng.dates[i], self.dd[i], amt))
        if eng.cash > 1000 and not any(isinstance(o, Sell) for o in out):
            out.append(MOC(eng.cash, "invest"))
        return out

    def after_close(self, eng, i, filled):
        if self.payback and eng.debt > 0:
            eng.repay(eng.debt, i)
        if self.payback and (eng.debt <= 1e-6 or any(t.get("tag") == "margin_call" for t in filled)):
            self.payback = False
        gross = eng.shares * eng.c[i] + eng.cash
        self.ltv.append(eng.debt / gross if gross > 0 else 0.0)
        if self.dd[i] >= -1e-12:
            if self.repay_mode == "recovery" and eng.debt > 0:
                self.payback = True
            self.armed = True
        elif self.armed and self.dd[i] <= -self.trigger:
            self.armed, self.go = False, True
