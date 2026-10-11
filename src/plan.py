"""
src/plan.py  -  study 13: Matt's own plan. Rising contributions + 1.25x-1.5x leverage, two ways.

One small simulator, vectorised over many start dates at once (W windows x T steps).
Each step: markets move -> loan interest -> margin check -> (contribution day) trade at the close.

TWO ways to hold more than 1x:
  'margin'  one 1x fund + a loan. On every contribution day the account is set back to the target
            exposure L x net worth: borrow and buy after gains, SELL (26% tax on the gain at average
            cost) and repay after losses. Margin call when equity < 25% of the position: scenario
            ends at ZERO (Matt's rule from study 09).
  'etf'     a 1x fund + a daily 2x fund, no loan. Target weight in the 2x fund = L - 1
            (75/25 -> 1.25x, 50/50 -> 1.5x). NEVER sells: each contribution goes to the
            underweight fund first (Italian tax: ETF losses can't offset ETF gains, so selling
            to rebalance only ever costs tax). Exposure drifts between contributions.
Costs: FEE on every buy and sell. End: everything sold, 26% on each fund's gain separately
(a loss on one fund does not offset the gain on the other), loan repaid.
"""
import numpy as np

FEE, CGT, MM = 0.001, 0.26, 0.25


# ------------------------------------------------------------------ contribution plans (EUR / quarter)
def plan_matt(k):
    """k = quarter number from the start (0, 1, 2...). Written down BEFORE running (decisions.md)."""
    k = np.asarray(k)
    full = 3000.0 * 1.05 ** ((k - 16) // 4)
    return np.where(k < 4, 500.0, np.where(k < 16, 1200.0, full))


def plan_flat(k):
    return np.full(np.shape(k), 500.0)


# ------------------------------------------------------------------ leverage paths (target exposure by quarter)
def const(L):
    return lambda k: np.full(np.shape(k), float(L))


def glide(k):
    """1.5x for the first 5 years (20 quarters), then 1.25x."""
    return np.where(np.asarray(k) < 20, 1.5, 1.25)


# ------------------------------------------------------------------ simulator
def simulate(r1, r2, mrate, contrib, target, mode):
    """
    r1      (W,T) return of the 1x fund over step t (total return)
    r2      (W,T) return of the 2x fund over step t (ignored in 'margin' mode)
    mrate   (W,T) loan interest charged over step t (fraction, already per step)
    contrib (W,T) EUR paid in at the close of step t (0 = no contribution that step)
    target  (W,T) target exposure on contribution steps
    Returns a dict of arrays.
    """
    r1, r2, mrate, contrib, target = (np.asarray(a, float) for a in (r1, r2, mrate, contrib, target))
    W, T = r1.shape
    s1, s2, b1, b2, D = (np.zeros(W) for _ in range(5))
    alive = np.ones(W, bool)
    bust_step = np.full(W, -1)
    residual = np.full(W, np.nan)
    interest, tax_paid, fees = np.zeros(W), np.zeros(W), np.zeros(W)
    nw_pre, nw_post, expo = (np.zeros((W, T)) for _ in range(3))
    for t in range(T):
        s1 *= 1 + r1[:, t]
        if mode == "etf":
            s2 = np.maximum(s2 * (1 + r2[:, t]), 0.0)
        it = D * mrate[:, t]
        D += it
        interest += it
        call = alive & (D > 0) & (s1 + s2 - D < MM * (s1 + s2))
        if call.any():                                   # margin call: forced sale, scenario = zero
            residual[call] = (s1 + s2 - D)[call]
            bust_step[call] = t
            alive &= ~call
            for a in (s1, s2, b1, b2, D):
                a[call] = 0.0
        nw_pre[:, t] = s1 + s2 - D
        c = np.where(alive, contrib[:, t], 0.0)
        go = c > 0
        if go.any():
            L = target[:, t]
            if mode == "margin":
                need = np.where(go, L * (s1 - D + c) - s1, 0.0)
                buy = np.maximum(need, 0.0)
                sell = np.minimum(np.maximum(-need, 0.0), s1)
                frac = np.divide(sell, s1, out=np.zeros(W), where=s1 > 0)
                gain = sell - b1 * frac
                tax = CGT * np.maximum(gain, 0.0)
                b1 = b1 * (1 - frac) + buy
                s1 = s1 - sell + buy * (1 - FEE)
                proceeds = sell * (1 - FEE) - tax
                tax_paid += tax
                fees += FEE * (buy + sell)
                D = D - (c - buy + proceeds)
                extra = np.maximum(-D, 0.0)              # can't happen for L >= 1; kept safe
                D = np.maximum(D, 0.0)
                s1 += extra * (1 - FEE)
                b1 += extra
            else:
                w2 = np.clip(L - 1.0, 0.0, 1.0)
                tot = s1 + s2 + c
                buy2 = np.clip(w2 * tot - s2, 0.0, c)
                buy1 = c - buy2
                s1 += buy1 * (1 - FEE)
                s2 += buy2 * (1 - FEE)
                b1 += buy1
                b2 += buy2
                fees += FEE * c
        nw_post[:, t] = s1 + s2 - D
        nw = nw_post[:, t]
        expo[:, t] = np.where(nw > 0, (s1 + 2 * s2) / np.where(nw > 0, nw, 1.0), np.nan)
    final_before = np.where(alive, s1 + s2 - D, 0.0)
    end_tax = np.where(alive, CGT * (np.maximum(s1 - b1, 0) + np.maximum(s2 - b2, 0)), 0.0)
    return {"final_before": final_before, "final_after": final_before - end_tax, "end_tax": end_tax,
            "alive": alive, "bust_step": bust_step, "residual": residual, "interest": interest,
            "tax_on_rebalancing": tax_paid, "fees": fees, "nw_pre": nw_pre, "nw_post": nw_post,
            "exposure": expo, "contributed": contrib.sum(axis=1)}


def drawdown_stats(nw_pre, nw_post, steps_per_year):
    """Time-weighted index of the portfolio (contributions removed). Returns max drawdown and the
    longest stretch below a previous peak (years). A margin call = -100%, underwater to the end."""
    W, T = nw_pre.shape
    prev = np.concatenate([np.zeros((W, 1)), nw_post[:, :-1]], axis=1)
    r = np.divide(nw_pre, prev, out=np.ones((W, T)), where=prev > 0) - 1
    dead = (prev > 0) & (nw_pre <= 0)
    r = np.where(dead, -1.0, r)
    idx = np.cumprod(1 + np.maximum(r, -1.0), axis=1)
    peak = np.maximum.accumulate(idx, axis=1)
    dd = np.where(peak > 0, idx / np.where(peak > 0, peak, 1) - 1, -1.0)
    under = dd < -1e-9
    longest = np.zeros(W)
    for w in range(W):
        run = best = 0
        for u in under[w]:
            run = run + 1 if u else 0
            best = max(best, run)
        longest[w] = best / steps_per_year
    return dd.min(axis=1), longest
