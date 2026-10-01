# Decisions log

- 2026-09-23: project created, config.yaml v1.

- 2026-09-30: Added descriptive reference line "perfect-timing ceiling" (hindsight: each quarter's cash invested at that quarter's lowest close). Not a strategy, never selectable. Used to report ceiling_share. Added metric avg_wait_days (money-weighted days a euro waited in cash). Neither changes any pre-registered rule.

- 2026-10-01 (study 07, before running it): interpretation choices for §9 where the prereg was not explicit:
  (a) "deadline one step" uses the order 30 < 60 < next < none (by longest possible wait);
  (b) bootstrap histories use price bars only, no dividends, for both the strategy and Buy & Hold;
  (c) PBO performance per block = mean daily time-weighted return in excess of Buy & Hold;
  (d) finalist types are A, B, C as written; daily-reference rules ("-") have no re-entry type and are reported but not selectable;
  (e) bug fix: avg_wait_days now counts never-invested cash as waiting until the last day (XIRR results unaffected; grid re-run).