# ES Entry System

Does a rules-based entry system (dip + trend + index valuation + sizing) beat
Buy & Hold and monthly DCA on the same EUR 500/quarter, buying IUSA?

Rules: no leverage, no look-ahead, no parameter fitting, raw data read-only.
Pipeline: scripts/0_fetch_data.py -> 1_build_daily.py -> 2_engine.py ->
3_strategies.py -> 4_metrics.py -> 5_report.py -> 6_robustness.py
