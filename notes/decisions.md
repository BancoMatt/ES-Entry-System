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