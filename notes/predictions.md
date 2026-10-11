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


13. I believe that (before reading any result) lump sum B&H once a year will beat quarterly B&H and all of the other strategies, probably just thanks to timing during rough years, and lower cash drag compared to DCA, buy the dip etc...

14. before running 9_holdout.py, I believe that quarterly B&H will beat monthly DCA, mostly due to lower cashdrag and lower fees (less entries), I would say that about half of the configs beat B&H (guess), quarterly B&H beats yearly B&H though they will be close, in my opinion.

15. For the test of what to do if a crash happens while applying the quarterly B&H strategy, I predict that selling to invest in the dip loses money, waiting out loses money, building a cash deposit to deploy during crashes loses money to cash drag and would be better off invested together with the quarterly B&H, the only way that you can maximize profits during crashes is investing the upcoming quarterly B&H all at once or layer entries at -10%/-20%/-30% from top. Front loading though needs an emergency fund and money actually available, so there's that to account for, if you don't have the money physically you can't invest it.

16. before seeing results, in the ACC vs DIST test, I believe that the ACC will outperform DIST of about half a point per year, probably the only years when it will pay off less will be specific multi year bear market time frames, but zooming out to a larger time frame, it will actually perform better because it will automatically reinvest dividends at lows that would have else been skipped by dividend reinvesting and would have sat in cash and used elsewhere.

17. on the 1871 to 1998 front load test, I've decided to run it again on a way longer data frame on data it had never seen (excluding 1999-2026) because I wanted to test wether the -15% row would survive new data.
I think it will stay around the same, still in the "lucky" range, perhaps decay a bit due to 1871-1998 having crashes that are years long, normal B&H would average out the entries better, front loading will probably get stuck at intermediate levels that will yield worse returns. I do think though that the performance of slightly front loading is worth it on smaller crashes, though you need to be able to judge if a crash is a flash one rather than a decade long one, that has to do with your macroeconomical/geopolitical judgment and can't really be foreseen.