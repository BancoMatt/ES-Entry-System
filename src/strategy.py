"""
src/strategy.py  -  the dip strategy as a Policy the engine can run.

Same rules as src/signals.py (reference, drop size, re-entry A/B/C), but CASH-AWARE,
as the pre-registration (§5-6) requires:
  - money is split into PIECES: one piece = (quarterly contribution) / P
  - every buy spends one piece (ladder: one piece per level reached)
  - if even 1 share isn't affordable, NO order is placed and the rule's memory does
    not change (it waits for the next signal)
  - DEADLINE: euros that waited >= D days get invested at the close
    ("next" = leftover cash is invested on the day the next contribution arrives)

E1 (intraday limit): at the open, place limit orders at ref x (1 - x). The engine fills
    them if the day's low goes through (touch, or 0.05% trade-through).
E2 (end of day):     after the close, check close <= ref x (1 - x). If so, buy at the
    NEXT day's close (market-on-close).

The rule's memory (armed / last buy / ladder level) only changes when an order FILLS.
"""
import numpy as np

from src.engine import MOC, Limit, Policy
from src.pit import known_at_open_features
from src.signals import OPEN_RELIABLE_FROM, Rule

DEADLINES = ("none", 30, 60, "next")


class DipStrategy(Policy):
    def __init__(self, rule: Rule, style="E1", pieces=2, deadline="none", trade_through=0.0005):
        if style == "E2" and rule.ref == "open":
            raise ValueError("'open' reference not available in E2")
        if deadline not in DEADLINES:
            raise ValueError(deadline)
        self.rule, self.style, self.pieces = rule, style, pieces
        self.deadline, self.tt = deadline, (trade_through if style == "E1" else 0.0)
        self.name = f"{style}|{rule.name}|P{pieces}|D{deadline}|tt{self.tt*1e4:g}bp"

    # ------------------------------------------------------------ setup
    def setup(self, eng):
        f = known_at_open_features(eng.px_all).loc[eng.dates]
        r = self.rule
        self.x = (np.full(len(eng.dates), r.thr) if r.thr_kind == "fixed"
                  else r.thr * f["vol_20d"].to_numpy(float))
        if r.ref == "open":
            ref = eng.o.copy()
            ref[eng.dates < OPEN_RELIABLE_FROM] = np.nan
        elif r.ref == "prev_close":
            ref = f["prev_close"].to_numpy(float)
        else:
            ref = f[r.ref].to_numpy(float)
        self.ref = ref
        self.piece = eng.per_year / 4 / self.pieces
        self.contrib_idx = {eng.dates.get_loc(d) for d in eng.contrib}
        # rule memory
        self.armed, self.last_fire = True, -10**9
        self.fired, self.anchor, self.x_ep = 0, np.nan, np.nan
        self.pending = None            # E2: (levels, anchor, x) waiting for tomorrow's close
        self.placed = []               # E1: what was placed today, to update memory tonight

    # ------------------------------------------------------------ helpers
    def _can_afford(self, eng, price):
        return eng.cost.affordable(min(self.piece, eng.cash), price) >= 1

    def _levels(self, i):
        """Limit/trigger levels in force today (before looking at today's prices)."""
        r, ref, x = self.rule, self.ref[i], self.x[i]
        if np.isnan(ref) or np.isnan(x):
            return [], None, None
        if r.reentry == "-":
            return [ref * (1 - x)], None, None
        if r.reentry == "A":
            return ([ref * (1 - x)] if self.armed else []), None, None
        if r.reentry == "C":
            return ([ref * (1 - x)] if i - self.last_fire >= r.cooldown else []), None, None
        # B ladder
        a, xe = (ref, x) if self.fired == 0 else (self.anchor, self.x_ep)
        return [a * (1 - (k + 1) * xe) for k in range(self.fired, r.ladder_max)], a, xe

    def _record_fill(self, i, n_filled, a, xe):
        r = self.rule
        if n_filled <= 0:
            return
        if r.reentry == "A":
            self.armed = False
        elif r.reentry == "C":
            self.last_fire = i
        elif r.reentry == "B":
            if self.fired == 0:
                self.anchor, self.x_ep = a, xe
            self.fired += n_filled

    def _rearm_on_new_high(self, eng, i):
        if self.rule.reentry in ("A", "B") and not np.isnan(self.ref[i]) and eng.c[i] >= self.ref[i]:
            self.armed, self.fired = True, 0

    # ------------------------------------------------------------ E1: open
    def on_open(self, eng, i):
        self.placed = []
        if self.style != "E1":
            return []
        levels, a, xe = self._levels(i)
        orders = []
        for lv in levels:
            if not self._can_afford(eng, min(eng.o[i], lv)):
                break
            orders.append(Limit(lv, self.piece, self.tt, "dip"))
        self.placed = (orders, a, xe)
        return orders

    # ------------------------------------------------------------ close (both)
    def on_close(self, eng, i):
        orders = []
        if self.style == "E2" and self.pending is not None:
            k = self.pending[0]
            orders.append(MOC(k * self.piece, "dip"))
        dl = self._deadline_budget(eng, i)
        if dl > 0:
            orders.append(MOC(dl, "deadline"))
        return orders

    def _deadline_budget(self, eng, i):
        if self.deadline == "none" or not eng.lots:
            return 0.0
        today = eng.dates[i]
        if self.deadline == "next":
            if i not in self.contrib_idx:
                return 0.0
            return sum(a for d, a in eng.lots if d < today)
        return sum(a for d, a in eng.lots if (today - d).days >= self.deadline)

    # ------------------------------------------------------------ evening (both)
    def after_close(self, eng, i, filled):
        dip_fills = [t for t in filled if t["tag"] == "dip"]
        if self.style == "E1":
            if self.placed:
                _, a, xe = self.placed
                self._record_fill(i, len(dip_fills), a, xe)
        else:
            if self.pending is not None:              # yesterday's signal: did it fill today?
                k, a, xe = self.pending
                if dip_fills:
                    self._record_fill(i - 1, k, a, xe)
                self.pending = None
            levels, a, xe = self._levels(i)
            k = sum(1 for lv in levels if eng.c[i] <= lv)
            if self.rule.reentry == "B":               # ladder: only consecutive levels count
                k = next((j for j, lv in enumerate(levels) if eng.c[i] > lv), len(levels))
            if k > 0 and i + 1 < len(eng.dates) and self._can_afford(eng, eng.c[i]):
                self.pending = (k, a, xe)
        self._rearm_on_new_high(eng, i)
