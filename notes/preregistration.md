# Pre-registration: ES Entry System

**Status:** FINAL v1, 2026-09-24. Reviewed by Matt: points 1, 2, 5, 8 confirmed or adjusted; finalist rule changed to one per re-entry type (see §9) after the candidate lists showed C20 dominating by construction. Frozen once it's committed and tagged `prereg-v1` in git. No profit or return figure has been computed yet. Only data checks and signal *counts* exist so far.

**The rule of this document:** everything here was decided *before* seeing any result. If something has to change later, the change goes into `notes/decisions.md` with the date and the reason, and every result produced after it is labelled `post-prereg`.

---

## Plain-English summary

We test whether buying IUSA on dips beats simply investing each €500 when it arrives. Timing rules are fixed, costs are real, the tuning is done on old data and the check is done on new data. We try many settings on purpose. A winner only counts if a whole neighbourhood of similar settings also wins (a plateau), and if it still wins on 2019–2026 data it has never seen. If nothing passes, the answer is "just invest on arrival", and that's a valid result.

---

## 1. Questions

- **Q1 (main):** Does any drawdown-based entry rule beat **Buy & Hold on arrival** (B&H) on money-weighted return (XIRR) after all costs, using Matt's real setup (€500/quarter, IBKR, IUSA)?
- **Q2:** Does it beat **monthly DCA**, both on the first trading day of the month and across every day-of-month?
- **Q3 (secondary, exploratory):** how do the answers change with fill model, fees, budget size, funding availability, Dist vs Acc, trading hours, leverage, and valuation/trend overlays?

**Why B&H is the main benchmark:** with quarterly cash, B&H is what Matt would do by default. A rule that beats monthly DCA but loses to B&H isn't worth using.

**Expected answer (stated in advance):** most rules lose to B&H on XIRR because money waits in cash. We expect at best a small edge, and mainly in drawdown and crash-buying, not in total return.

## 2. Data and periods

| Item | Setting |
|---|---|
| Price series | `pit_prices.parquet`. Eras: synthetic 1999-01-04 → 2002-03-14, fund NAV 2002-03-15 → 2008-12-31, real IUSA 2009-01-02 → today |
| Dividends | `pit_distributions.parquet`. Cash credited on the **pay date**, amount × shares held on the ex-date, minus 26% tax |
| Slow data | Shiller with a 6-month lag. 10y yield and FX from the previous day |
| **E1 build period** | **2009-01-02 → 2018-12-31** (real IUSA only; the pre-2009 proxy failed the check in study 03) |
| **E2 build period** | **1999-01-04 → 2018-12-31** (uses closes only, which are real in every era) |
| **Holdout (both)** | **2019-01-01 → last available day**. Opened **once**, with `--holdout`, and every opening is logged in `notes/holdout_log.md` |
| E1 on 1999–2008 | Reported as "indicative (proxy)" only. Never used for selection |

Known limitation: the E1 build period (2009–2018) has no bear market. E2's includes 2000–02 and 2008.

## 3. Account model (baseline)

| Setting | Baseline value |
|---|---|
| Contributions | €500 on the first trading day of Jan / Apr / Jul / Oct. Cash stacks up if unused |
| Instrument | IUSA (Dist), Euronext Amsterdam, EUR |
| Leverage | none. Cash can never go below 0 |
| Shares | **whole shares only** (conservative). Fractional shares are a sensitivity in study 04 |
| Fee per order | `min(29, max(1.25, 0.05% × value)) + 0.30` (IBKR Tiered + estimated exchange/clearing) |
| Spread | half-spread of 2.5 bps added to market / market-on-close orders. Limit orders pay their limit (or a better open) |
| Idle cash interest | 0 |
| Dividend tax | 26% withheld when the dividend is credited |
| Valuation | shares × close + cash, every day |

## 4. Benchmarks

- **B&H on arrival:** every cash inflow (contribution or dividend) is invested at that day's close (market-on-close) in whole shares. Leftover euros stay in cash and get added to the next buy.
- **Monthly DCA:** €166.67 on day *d* of each month (the first trading day on or after *d*). Money still arrives quarterly, so it waits in cash until its month. Headline: *d* = first trading day. We also run every *d* = 1…28 and show the distribution.
- **Quarterly DCA** = B&H, and is shown as such.

## 5. Entry rules (the signal grid)

Reference prices (all known before the decision):
- `high_Nd` = highest close of the previous N days, N ∈ {5, 10, 20, 50}
- `prev_close` (daily reference, run for completeness)
- `open` (E1 only; open prices only exist from 2007)

Drop size *x*:
- **fixed** ∈ {0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5}%
- **volatility-scaled** *x* = k × 20-day volatility, k ∈ {1, 1.5, 2, 2.5, 3}

Rules for buying more than once within the same drop (`high_Nd` only):
- **A:** one buy per drop. Re-armed only by a close at a new N-day high.
- **B:** a ladder, with one buy at each deeper step (x, 2x, 3x … max 5 steps) below the high where the drop started. Everything resets at a new high.
- **C5 / C20:** another buy is allowed whenever the condition holds and ≥ 5 / 20 trading days have passed since the last buy.

Grid size:
- **E1:** 4 × 15 × 4 = 240 drawdown rules, plus 30 daily rules = **270**
- **E2:** 240 drawdown rules, plus 15 `prev_close` rules = **255**

### Execution styles
- **E1, intraday limit:** at the open of day *t*, place a limit order at `ref × (1 − x)`.
  - Fill rule, primary (conservative): fills only if the day's low ≤ limit × (1 − 0.05%).
  - Fill rule, reported alongside: fills if the low touches the limit.
  - Fill price = min(open, limit). Never above the limit.
- **E2, end of day:** signal on day *t* if `close_t ≤ ref_t × (1 − x)`, with `ref` built from closes *before* *t*. Buy at the **close of day t+1** (market-on-close) + half-spread. The rule states (A/B/C) update on closes, same as E1.

### Candidate set, chosen by frequency (no returns involved)
A rule is a **candidate** if, over its own build period and execution style, it fires **8–12 times per year on average, and at least 2 times in its worst full year** (≈ 2–3 buys per quarter, the most €500/quarter can fund with sensible fees).
- The E1 list is recomputed on **2009–2018 real data only**. The provisional 1999–2018 run found 58 rules.
- The E2 list is computed with close-based signals on 1999–2018.
- Both lists are generated by script, saved as `notes/candidates_E1.csv` / `notes/candidates_E2.csv`, and committed **before** the engine runs.

## 6. Sizing and cash

- **Pieces P ∈ {1, 2, 3, 4}.** One piece (tranche) = €500 / P. Every fill spends one piece, or less if that's all the cash there is. With the ladder, k levels on the same day = k pieces.
- Order = the most whole shares affordable with the piece, after the fee. If not even 1 share is affordable, **no order is placed and the rule's state doesn't change**, so it waits for the next signal.
- **Deadline D ∈ {none, 30 days, 60 days, next contribution}.** Cash is tracked by date, oldest first. Any euros waiting ≥ D (calendar days) get invested at that day's close (market-on-close). "Next contribution" means leftover cash is invested at the close of the day the next €500 arrives.
- Dividends join the cash pool on their pay date and follow the same rules.

## 7. Run count (everything counts as a trial)

- E1: 270 rules × 4 P × 4 D × 2 fill rules = **8,640**
- E2: 255 × 4 × 4 = **4,080**

Total **12,720** configurations in the core study. We report this number wherever results appear.

## 8. Metrics

- **Primary: ΔXIRR vs B&H** (percentage points per year). XIRR runs on the actual cash flows (contributions on their dates, final value on the last day), in EUR, **before selling** (final value = shares × close + cash).
- Also reported:
  - XIRR after selling (minus 26% tax on the gain)
  - ΔXIRR vs monthly DCA, both headline and day-of-month distribution
  - final value and profit
  - **fees paid (€ and % of contributions)**
  - average % of money sitting in cash
  - max drawdown of account value
  - number of buys and the average price paid

## 9. How a winner is picked (plateau rule)

A configuration *c* = (execution, rule, P, D) **qualifies** if all of these hold on the build period, using the primary fill rule:

1. its rule is in the candidate set,
2. ΔXIRR vs B&H > 0,
3. ≥ 75% of its **neighbours** also have ΔXIRR > 0. Neighbours are the same configuration with the drop size one step up or down, P ± 1, or D one step either way.
4. ΔXIRR > 0 in **both halves** of its build period (E1: 2009–13 / 2014–18; E2: 1999–2008 / 2009–18),
5. in block-bootstrap Monte Carlo (500 alternative histories built from 63-day blocks of real daily bars), ΔXIRR > 0 in ≥ 70% of them.

**Picking finalists (at most 3 per execution style: one per re-entry type)**
- Group the qualifying configurations into families (reference, re-entry rule, fixed/vol).
- Split the families by **re-entry type**: A, B, and C (C5 and C20 together). At most **one finalist per type**, so ≤ 3 per execution style.
- Within each type, rank families by the **share** of their configurations that qualify (qualifying ÷ all configurations of that family), **not the raw count**.
  - Why: C20 families start with 3× more candidates than A or B (the 20-day cooldown caps buys at ~12/yr, so most drop sizes land in the frequency band by construction). Ranking by count would favour C20 for a reason that has nothing to do with performance.
  - Ties: the family with more qualifying configurations wins, then the simpler rule.
- In the chosen family, the finalist is the configuration whose ΔXIRR is closest to the family's **median**, never its maximum. Ties go to lower fees, then to the simpler rule.
- A type with no qualifying family sends no finalist.
- Strong runner-up families are shown in the heatmaps and discussed. They are **not** tested on the holdout, because each extra holdout test raises the chance of a lucky pass.

**The whole map is always shown, not just the winners.** Every configuration's neighbour share is reported as a heatmap. Colour bands:
- **≥ 75%:** qualifies
- **60–75%:** "near-plateau", shown and discussed, never selected
- **< 60%:** no plateau

A family that only reaches 65–70% is still visible and talked about. It just can't become a finalist.

The same plateau test is also run **vs monthly DCA** and reported as a secondary result (answers Q2).

**If nothing qualifies:** the pre-registered conclusion is "no evidence that these dip rules beat investing on arrival". For transparency, the family with the most configurations beating B&H still gets one run in the holdout, clearly labelled as not qualified.

**Overfitting check:** probability of backtest overfitting (CSCV, 8 blocks) on the candidate set, reported with the finalists.

## 10. Holdout protocol

1. Commit the finalists to `notes/finalists.md` **before** opening the holdout.
2. Run finalists + benchmarks once on 2019–today with `--holdout`. The unlock is logged automatically.
3. **Success:** a finalist's ΔXIRR vs B&H > 0 in the holdout. All finalists are reported, pass or fail.
4. Nothing gets changed after the holdout. Anything run afterwards is labelled `post-holdout`.

**When Matt would switch his real account:** only if a finalist passes the build selection **and** the holdout **and** stays positive after the fee stress test (IBKR Fixed, €3 minimum).

## 11. Secondary studies (exploratory, never used to pick finalists)

These are run on benchmarks + finalists. If there are no finalists, they use the labelled reference strategy from section 9. They're reported as descriptive.

| # | Study | Fixed settings |
|---|---|---|
| 03 | Fill model | touch vs trade-through 0.05% (E1) |
| 04 | Fees | IBKR Tiered (base) / IBKR Fixed €3 min / zero fees; whole vs fractional shares |
| 05 | Budget | €2k, 4k, 5k, 6k, 8k, 10k, 12k, 20k per year; piece size scales with budget |
| 06 | Funding availability | annual (€2,000 on the first trading day of Jan) / quarterly (base) / monthly (€166.67). Piece size stays €500/P |
| 07 | Dist vs Acc | IUSA vs the Acc fund (CSPX/SXR8), priced from the **official iShares NAV in EUR**. Yahoo's SXR8 first days are wrong: NAV-to-NAV 2010-05-19 → 2026-09-22 gives Acc 14.76%/yr vs Dist price 13.05%/yr vs Dist total return 14.64%/yr. Acc: no dividends, 26% only when sold. Shown before and after selling. E2 + benchmarks only (the NAV is a close only) |
| 08 | Trading hours | ES full-day vs Amsterdam-window dip signals, 2010+, as a hypothetical "24h instrument" |
| 09 | Leverage (curiosity) | daily reset L ∈ {1, 2, 3, 4, 5}: daily return = L × r − (L − 1) × (ECB deposit rate + 0.5%)/252 − 0.6%/252. Applied to B&H, DCA and finalists, 1999+ (E2 timing) |
| 10 | **Purpose** | Answers one question: do value investing and trend (EMA) **add anything**? Each overlay is reported as the **extra** ΔXIRR over the same setup without it. It's applied to (i) the finalists or reference strategy and (ii) **plain monthly DCA** ("value-sized DCA": monthly amount × multiplier, limited by the cash available; unspent cash waits). Part (ii) tests value investing on its own, separate from any dip rule |
| 10a | Trend filter | buy only if the previous close is above its 200-day EMA, and the opposite (only below). Both variants |
| 10b | Value sizing | Matt's formula on the index: e = (P_N/P)^(1/N) − 1 + d, with EPS = lagged Shiller E, g = 10y EPS growth capped to [0%, 10%], M = median P/E of the prior 10 years, r = 10y yield + 5% (fixed), N = 10. Piece multiplier: × 1.5 if e − r ≥ +1%, × 1 in between, × 0.5 if e − r ≤ −1% |

## 12. Engineering commitments

- One engine for every study. Every run gets written to one master results table.
- Automatic tests: look-ahead (existing), plus accounting (cash never below 0; contributions + gains = value + fees + taxes; fees match the formula).
- The engine refuses to read dates after 2018-12-31 unless `--holdout` is given.
- This file, both candidate lists and the code get committed and tagged `prereg-v1` before the first engine run.
