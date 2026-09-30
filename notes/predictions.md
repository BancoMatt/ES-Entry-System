# Predictions (written BEFORE any strategy result)

Committed before `scripts/6_core_grid.py` was run for the first time.
Known at this point: only the benchmarks (study 02).

## Known facts at time of writing (study 02)
- 1999-2018 (E2 build): Buy & Hold 7.18%/yr XIRR before selling. Monthly DCA 7.05-7.09% on every day of the month (0 of 28 beat B&H, median -0.12 pts/yr).
- 2009-2018 (E1 build): Buy & Hold 12.53%. DCA 11.97-12.19% (0 of 28 beat B&H, median -0.48 pts/yr).
- DCA fees 0.93% of contributions vs 0.31% for B&H; DCA average cash 2.2-3.2% vs ~0.1%.

## Rule of thumb used to reason about results
required discount per euro ≈ mu x (W / 365) + extra fee %
- mu = market return per year, W = average days a euro waits in cash, extra fee = fee % above B&H's 0.31%
- per-euro loss -> XIRR gap ≈ loss / average years the money stays invested (~10 in E2, ~5 in E1)
- check: E2 DCA, W ~30 days, mu 7% -> 0.58% + 0.62% = 1.2% per euro / 10 ≈ 0.12 pts/yr = measured 0.12

## Claude's predictions
1. Most of the 12,720 configurations lose to Buy & Hold; median around -0.3 to -0.8 pts/yr.
2. E1 does worse than E2 (the 2009-2018 bull market makes waiting expensive).
3. More pieces lose more; "no deadline" loses most.
4. If anything qualifies, it is E2 and crash-driven (deep B ladders or large drops from 20/50-day highs), winning mainly thanks to 2002 and 2008.
5. The best configurations capture at most ~20-30% of the perfect-timing ceiling.
6. EMA and value overlays (study 10) add little or nothing.

## Matt's predictions
1. Will anything beat Buy & Hold? (yes / no, how many roughly?)
2. E1 or E2 better?
3. Which re-entry type (A / B / C) does best?
4. Which reference (5 / 10 / 20 / 50-day high) does best?
5. How many pieces work best (1 / 2 / 3 / 4)?
6. Does the deadline help?
7. Anything else you expect:
8. Most of the parameters will most likely fail against B&H.
9. The trending section (2009-2018) will perform worse than the whole period from 1999 and lots of cash drag will take place, and many entries will take place only because of the 30-60 days max cash drag rule.
10. The dip strategy will get to about half of the perfect timing ceiling (probably even less, about 30%) during major crashes like covid, 2008, 2000s, the rest of the time it will have higher entries than B&H.
11. more confluences will probably just create less entries and worse ones, leading to more cash drag and worse overall performance.
12. the strategy will be positive but just won't perform against B&H.