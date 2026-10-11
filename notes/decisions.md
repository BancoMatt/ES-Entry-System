# Decisions log

- 2026-09-23: project created, config.yaml v1.

- 2026-09-30: Added descriptive reference line "perfect-timing ceiling" (hindsight: each quarter's cash invested at that quarter's lowest close). Not a strategy, never selectable. Used to report ceiling_share. Added metric avg_wait_days (money-weighted days a euro waited in cash). Neither changes any pre-registered rule.

- 2026-10-01 (study 07, before running it): interpretation choices for §9 where the prereg was not explicit:
  (a) "deadline one step" uses the order 30 < 60 < next < none (by longest possible wait);
  (b) bootstrap histories use price bars only, no dividends, for both the strategy and Buy & Hold;
  (c) PBO performance per block = mean daily time-weighted return in excess of Buy & Hold;
  (d) finalist types are A, B, C as written; daily-reference rules ("-") have no re-entry type and are reported but not selectable;
  (e) bug fix: avg_wait_days now counts never-invested cash as waiting until the last day (XIRR results unaffected; grid re-run).

- 2026-10-10 amendment (before opening): the holdout also shows the yearly hindsight line (yearly money at each year's lowest close) next to yearly B&H. Descriptive only.

## 2026-10-10: study 06a pulled forward, calendar curiosity, holdout scope (written BEFORE opening the holdout)

1. **Study 06 (funding), benchmark part, run now on the build periods.** Pre-registered in §11
   (annual / quarterly / monthly arrival). Strategies compared within each arrival schedule:
   - yearly money (EUR 2,000, first trading day of Jan): lump sum on arrival, spread over 4 quarters,
     spread over 12 months, hindsight ceiling (year's lowest close)
   - quarterly money (EUR 500, base): B&H on arrival, save up and invest once a year in January,
     monthly DCA, hindsight ceiling
   - monthly money (EUR 166.67): B&H on arrival
   Exploratory and descriptive. Never used to pick or change finalists.

2. **Calendar curiosity (NOT pre-registered, descriptive only).** E2 build period (1999-2018), EUR 2,000
   once a year. Scan of every trading day k of the year in two versions ("arrives on day k" = pure
   calendar effect; "arrives in January, waits until day k" = cost of waiting), plus a Monte Carlo of
   10,000 paths with a random trading day each year (seed 2026). The best and worst fixed days are
   hindsight picks among ~250 tries and never become a rule.

3. **Holdout scope, fixed now.** One run with `--holdout` on 2019-01-01 to the last day, containing:
   - quarterly B&H (base), annual lump sum (yearly money), save-up annual, monthly DCA, quarterly ceiling
   - the two NOT QUALIFIED reference configs from notes/finalists.md
   - the full grid (12,720 configs) reported as a **distribution only** (share beating B&H, median,
     fig01-style histogram). No configuration is selected, ranked or promoted from the holdout grid.
   Success criterion unchanged (§10). Anything run after the holdout is labelled `post-holdout`
   (e.g. repeating the calendar curiosity on 1999 to today).

2026-10-10: study 07 Dist vs Acc (POST-HOLDOUT, exploratory; written BEFORE running)
Script: scripts/10_dist_vs_acc.py. Data via src/data.py load_post_holdout() (full history, locked until notes/holdout_results.md exists; not logged as a holdout opening).
Funds: IUSA (Dist) vs CSPX/SXR8 (Acc), both priced from their OFFICIAL NAV in EUR (ECB fix) on the same days. Official period = Acc launch 2010-05-19 to the last NAV day. Also Dist priced from the IUSA market close, as a consistency check.
Strategies: quarterly B&H (main), monthly DCA, E2 reference config. EUR 500/quarter, IBKR Tiered, whole shares. Dist: 26% tax on each dividend at payment. Acc: no tax until sale. Both: 26% on the gain if sold. IVAFE (0.2%/yr, same for both) ignored.
Split of the gap (keep holding): dividend tax drag = Dist with no dividend tax - Dist; fund level + reinvestment delay = Acc - Dist with no dividend tax.
Holding period: rolling windows, a new start each quarter, 5/10/15/20 years, sold at the end. Before 2010-05-19 the Acc price is the Dist fund's official total-return index (EUR), spliced at launch; those windows are marked "proxy".
Accounting fix for this study: dividends already earned (ex-date passed) but not yet paid are counted in the Dist account value. The engine credits them on the pay date; that is neutral when every policy holds the same fund (all earlier studies) but would bias a Dist vs Acc comparison.

2026-10-10: study 11 crash playbook (POST-HOLDOUT, exploratory; written BEFORE running)

Question: I invest EUR 500 every quarter (quarterly B&H). A crash comes. What should I do? Script: scripts/11_crash_playbook.py, policies in src/crash.py. Data via load_post_holdout(). IUSA (Dist), whole shares, IBKR Tiered, as in the main study. Scored on XIRR AFTER selling at the end vs plain quarterly B&H on the same period. Periods: 1999-2026, plus separate starts 1999-2008, 2009-2018, 2019-2026. Drawdown = close vs the highest close so far (all history). Decided at the close of day t, acted on at the close of day t+1.

Engine additions (all off unless used; tests/test_engine_debt_sell.py): Sell orders (fee, 26% tax on the realised gain at average cost, no credit for losses), debt (borrow, repay, repay-from-contributions, daily interest at a rate series), final value net of debt, full accounting identity checked on every run.

A behaviour: pause buying at -20%, restart at -10% / at a new high; panic-sell everything at -20%, buy back at -10% / at a new high. B front-loading: at -X% (10/15/20/25/30/40) invest the next N quarters (1/2/4/8) now, borrowed from the emergency fund, repaid by the next contributions. One-shot per crash (re-armed at a new high) or ladder (another N at every further -10%). Max 8 quarters outstanding. Emergency fund in plain cash (0%) or in a deposit paying the ECB deposit rate (floored at 0%), where the interest it would have earned counts as a cost. C placebo: every front-load config (1999-2026, plain cash) re-run 200x with the same number of front-loads on random days (seed 2026). Reported as the percentile of random runs beaten. R reserve control line: 10% of each contribution kept in cash, all invested at -20%. D credit line, big investor: EUR 1M lump sum, no salary, dividends reinvested above EUR 1,000 cash. At -X% (15/20/30) borrow f (10/20%) of net worth and buy, only if the loan stays <= 30% of the portfolio; interest = ECB deposit rate + 1.5% (floored at 0). Repay at the next all-time high by selling (taxed), or carry the loan to the end. Margin call: if the loan exceeds 50% of the portfolio, forced sale back to 40%.

Representative configs, fixed now for the per-crash tables and charts: front-load -20%, N = 4, one-shot, plain cash | credit line -20%, borrow 20% (both repay modes).

2026-10-11: study 11b, out-of-sample check of the -15% front-load (written BEFORE running)

Why: in study 11 the -15% trigger was the only blue row in fig19 (beats 70-88% of random dates). It was noticed AFTER seeing the data, so it is tested on data it never saw.

Data: Shiller monthly S&P 500, 1871-01 to 1998-12 (before our 1999+ sample). USD.
Same rules as study 11 where possible: USD 500 per quarter (Jan/Apr/Jul/Oct), drawdown vs the highest price so far, front-load at -15%, N = 1/2/4/8 quarters, borrowed from the emergency fund at 0%, repaid by skipping the next contributions, one-shot per crash (re-arm at a new high), max 8 quarters outstanding. Decided on month t, invested at month t+1.
Placebo: same number of front-loads on random months, 200 per size, seed 2026.
Simplifications (identical for every line): Shiller P = monthly AVERAGE price, dividends D/12 reinvested monthly, 0.1% cost per buy, no taxes, contributions fixed at USD 500 nominal.
SUCCESS (pre-registered): the crash-timed version beats >= 90% of random dates in at least 3 of the 4 sizes over 1871-1998. Otherwise the study 11 result is treated as luck.

2026-10-11: study 10 trend + value overlays (POST-HOLDOUT, exploratory; written BEFORE running)

Script: scripts/12_overlays.py, overlays in src/overlays.py (tests/test_overlays.py, incl. a look-ahead test of the value signal). Data via load_post_holdout(). IUSA (Dist), whole shares, IBKR Tiered, EUR 500/quarter, as in the main study.

Pre-registered (§11): each overlay is reported as the EXTRA XIRR over the same setup without it.

10a trend: buy only if yesterday's close > its 200-day EMA; and the opposite (only below). A skipped buy leaves the cash waiting for the next buy day that qualifies.
10b value sizing, Matt's formula on the S&P 500 (USD): EPS = Shiller E with 6-month lag, g = 10y EPS growth capped [0%, 10%], M = median P/E of the prior 120 months (lagged), P = S&P close of the previous day, d = lagged Shiller dividends / P, e = (EPS x (1+g)^10 x M / P)^(1/10) - 1 + d, r = 10y US yield (previous close) + 5%. Order size x1.5 if e - r >= +1%, x1 in between, x0.5 if e - r <= -1%. The unspent half goes into a saved pot; the extra half when cheap comes only from that pot.
Bases: monthly DCA (pre-registered) and the E2 reference dip rule (pre-registered "reference strategy"). ADDED, not pre-registered: quarterly B&H, the main benchmark. E1 reference skipped: intraday data before 2009 isn't reliable and limit orders don't map onto the value pot.
Periods: 1999-2026 (main), plus separate starts 1999-2008, 2009-2018, 2019-2026 (same end/start caveat as study 11).
Descriptive check of the signals themselves: IUSA total return (EUR) over the next 1 and 3 years after each day, grouped by signal state. Overlapping windows, so day counts overstate the number of independent observations.

2026-10-11: study 09 leverage (POST-HOLDOUT, curiosity; written BEFORE running)

Script: scripts/13_leverage.py, policies in src/leverage.py (tests/test_leverage.py). EUR 500/quarter in IUSA, IBKR Tiered. Three separate starts: 2000, 2010, 2020 -> last day. Leverage L: 1x (= quarterly B&H, the reference), 1.2, 1.5, 2, 3, 4, 5, 10. Matt's spec.

Model 1, margin loan (main):

Target leverage, re-set on every contribution day: borrow and buy after gains, sell (26% tax on any gain) and repay after losses. Idle cash (dividends, rounding) repays the loan.
Interest: ECB deposit rate + 1.5%, floored at 0. Whole shares.
Margin call when equity / position < min(25%, half the starting margin): 25% for 1.2-2x, 16.7% 3x, 12.5% 4x, 10% 5x, 5% 10x. Checked on every close.
Matt's rule: a margin call ends the scenario at ZERO (the forced sale's leftover is reported but counted as lost).
Reality note: a normal margin account allows about 2x; 3-4x needs portfolio margin; 5x and 10x are futures/CFD territory.

Model 2, daily-reset leveraged ETF (pre-registered §11):

daily return = L x r - (L-1) x (ECB rate + 0.5%)/252 - 0.6%/252, r = IUSA total return.
A day with L x r <= -100% wipes the fund out (ends at zero). Fractional shares (a decayed fund can trade below a cent). Quarterly B&H into the fund.

Outputs: final net worth, x vs 1x, XIRR before/after selling, max drawdown, bust date, interest paid, peak leverage reached. Charts: one panel per start, all leverages, X at the bust date.