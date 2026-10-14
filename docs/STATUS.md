# Build status

Tracks stage-by-stage completion, mirroring the other four engines'
own `docs/STATUS.md` convention.

## Final summary (all 10 stages complete)

**Architecture.** `regime/` (real market-wide features -> a
transparent, rule-based classifier) + `strategy_health/` (real
performance monitoring + real signal health, the latter built on top
of `factor-model-engine`'s own real `monitoring.compute_factor_stability`)
-> `regime_conditioned/` (strategy performance grouped by regime,
always marked as association, never causation) -> `diagnostics/` (five
real, separately-testable deterioration categories) -> `alerts/`
(severity-tiered promotion of already-computed real figures) ->
`pipeline/` + `reporting/` (one orchestration function, one
JSON/Markdown report). Depends on `factor-model-engine`,
`portfolio-construction-engine`, `portfolio-risk-engine`, and
`transaction-cost-engine` as pinned git dependencies; never re-runs an
optimizer, re-estimates a factor covariance, or re-prices a trade --
it only diagnoses what those engines' own real outputs already show.

**Important formulas:** market-wide regime features --
`realized_volatility` = trailing-window std of a real equal-weighted
cross-sectional return series; `trend_strength` = trailing cumulative
return of that same series; `avg_pairwise_correlation` = mean
off-diagonal entry of the real per-security return correlation matrix
over a trailing window; `liquidity_score` = relative change in
cross-sectional median ADV over a trailing window. Strategy
performance -- drawdown from the real wealth index
(`wealth = (1+net_returns).cumprod()`, `drawdown = wealth/wealth.cummax() - 1`);
rolling Sharpe via `alpha-validation-toolkit`'s own real
`sharpe_ratio`, applied per trailing window, never reimplemented.
Signal health -- `historical_ic_mean`/`recent_ic_mean`/`icir` read
directly from `factor-model-engine`'s own real
`FactorStabilityResult`; the real multi-horizon decay curve is the
same cross-sectional IC idea generalized across several forward
horizons using that engine's own `compute_factor_exposure_table`/
`_period_realized_returns`; exposure drift is the L1 change in a
portfolio's own real factor exposure vector between two consecutive
snapshots.

**Test counts.** 65 tests passing: 48 unit tests (full coverage of
every real classification rule path, drawdown/rolling-Sharpe
correctness against a known synthetic path, regime-grouping
correctness, every diagnostic category, every alert threshold),
14 integration tests on the real synthetic dataset (full pipeline,
reporting, reproducibility, point-in-time no-look-ahead, minimum-
history boundary), and 1 real CLI subprocess test (counted under
`slow`, still run by default).

**Reproducibility and point-in-time discipline, verified directly.**
This engine calls no numerical solver and no near-zero-magnitude BLAS
quadratic form of its own (unlike `portfolio-construction-engine`'s
SLSQP or `portfolio-risk-engine`'s risk decomposition), so exact
reproducibility is expected and confirmed directly, not just assumed.
The real point-in-time hardening check goes further than "don't pass
future dates": it confirms that handing the pipeline the FULL real
market/fundamental/optimization history (as a live system actually
would, with future rows already sitting in the feed) produces the
IDENTICAL result, at an earlier `as_of`, to handing it only the data
that existed by then -- every real PIT filter inside this pipeline and
every function it calls is responsible for this, confirmed rather than
trusted.

**End-to-end demonstration.** `scripts/run_regime_health_report.py`
generates a real synthetic dataset, runs a real monthly rebalancing
schedule through `portfolio-construction-engine`, then runs this
engine's complete Stage 2-8 analysis and prints the real regime label,
net return, drawdown, and every real alert raised.
`tests/integration/test_cli_workflow.py` runs it as a real subprocess.

**Limitations, identified rather than hidden:**
- Regime classification is a transparent, rule-based decision tree
  over five configured thresholds -- not a statistically fitted model
  (e.g. a hidden Markov model or Gaussian mixture). This is a
  deliberate design choice (transparency and auditability over
  statistical sophistication), documented in
  `docs/research_assumptions.md`, not a shortcut.
- The "signal" this engine's decay curve and exposure-drift diagnostics
  track is always one named factor's own exposure -- a composite,
  multi-factor alpha signal would need to be computed upstream and fed
  in as its own factor-like exposure series; this engine does not
  construct one itself.
- Regime-conditioned performance is explicit, labeled association, not
  causation, and the engine never claims otherwise -- but a researcher
  reading only the headline numbers out of context could still misread
  it that way; the disclaimer lives on the data object, not just in
  this document.
- All data is synthetic; this engine has no real-vendor data adapter
  of its own, and inherits every upstream limitation already
  documented in the four engines it depends on.
- Decay-curve computation recomputes `compute_factor_exposure_table` at
  every date for every configured horizon -- correct, but more
  compute than strictly necessary; acceptable for this engine's real
  point-in-time research scope, not optimized for a large universe or
  a long horizon list.

## Stage 1 -- Foundation: COMPLETE

Schemas (`data/schemas.py`): `RegimeFeatureRow`, `RegimeClassification`,
`StrategyPerformanceSnapshot`, `SignalHealthResult`,
`RegimeConditionedResult`, `DiagnosticFinding`, `Alert` -- six
structurally separate real concepts, matching this portfolio's own
convention of never merging distinct questions into one schema.
`EngineConfig`/`load_config` (`config.py`) with every real threshold
tagged RESEARCH ASSUMPTION. Dependency wiring to `factor-model-engine`,
`portfolio-construction-engine`, `portfolio-risk-engine`, and
`transaction-cost-engine` -- confirmed resolving cleanly through a
fresh venv install.

## Stage 2 -- Market Regime Features: COMPLETE

Added `regime/features.py` (`compute_regime_features`): real
volatility, trend, liquidity, and correlation features, built entirely
from `factor-model-engine`'s own real `_period_realized_returns`/
`latest_known_market` -- no return computation reimplemented. The
caller supplies a real point-in-time universe per date rather than
this module recomputing universe membership. Every feature is `None`
(never fabricated) when real history is insufficient, with an explicit
`quality_flags` entry recorded.

## Stage 3 -- Regime Classification: COMPLETE

Added `regime/classification.py` (`classify_regime`): a real,
transparent, rule-priority classifier. Every feature value is always
recorded in `evidence`, even when unused or missing (as `NaN`).
`UNKNOWN` (insufficient real history) and `AMBIGUOUS` (a feature
combination no rule recognizes, including a missing required feature)
are first-class real outcomes, never forced into the nearest label.

## Stage 4 -- Strategy Performance Monitoring: COMPLETE

Added `strategy_health/performance.py`
(`build_strategy_performance_snapshots`): gross/net return, turnover,
and transaction cost read directly from
`portfolio-construction-engine`'s own real `BacktestResult`/
`OptimizationResult` -- never recomputed. Drawdown derived from the
real net-return series via the standard wealth-index formula; rolling
Sharpe via `alpha-validation-toolkit`'s own real `sharpe_ratio`,
applied per real trailing window.

## Stage 5 -- Signal Health Monitoring: COMPLETE

Added `strategy_health/signal_health.py` (`compute_signal_health`).
Reuses `factor-model-engine`'s own real
`monitoring.compute_factor_stability` directly for
`historical_ic_mean`/`recent_ic_mean`/`icir`, including its own real
drift/sign-flip warnings, carried through unchanged rather than
re-derived with a second, parallel threshold. Adds two real
capabilities that module does not provide: a multi-horizon decay curve
(the same cross-sectional IC idea, generalized across forward
horizons) and portfolio-level exposure drift (the change in a specific
portfolio's own real factor exposure vector between two consecutive
snapshots).

## Stage 6 -- Regime-Conditioned Performance: COMPLETE

Added `regime_conditioned/conditioned_performance.py`
(`compute_regime_conditioned_performance`): real net-return
observations grouped by the real regime label classified for that
same date, with real descriptive statistics (mean, volatility, Sharpe
-- reusing `alpha-validation-toolkit`'s own `sharpe_ratio`) per regime.
`is_association_not_causation` is always `True` on the result object
itself, not just documented in prose.

## Stage 7 -- Diagnostic Framework: COMPLETE

Added `diagnostics/diagnostic_engine.py` (`run_diagnostics`): five
real, independently-testable functions --
`diagnose_data_quality` (wraps `factor-model-engine`'s own real
`DataHealthReport.warnings`), `diagnose_signal_deterioration` (wraps
`compute_factor_stability`'s own real warnings), `diagnose_exposure_drift`,
`diagnose_regime_change` (fires only on an actual real label change),
and `diagnose_cost_deterioration` (a real relative increase vs. the
real trailing average transaction cost). `is_causal_claim` is always
`False` on every `DiagnosticFinding`.

## Stage 8 -- Alerts: COMPLETE

Added `alerts/alerts.py` (`generate_alerts`): real, severity-tiered
thresholds applied to already-computed real figures (drawdown, rolling
Sharpe, recent IC, exposure drift, and promoted cost-deterioration
findings) -- mirrors `portfolio-risk-engine`'s own `limits.check_limits`
convention of applying NEW threshold comparisons without re-deriving
any underlying computation.

## Stage 9 -- Pipeline: COMPLETE

Added `pipeline/pipeline.py` (`run_full_regime_health_analysis`,
`RegimeHealthAnalysisResult`): the end-to-end orchestration of Stages
1-8 for one real portfolio's real rebalancing history, enforcing the
same point-in-time discipline (`as_of` must be the last real date in
`period_dates`) every other engine's own pipeline already enforces.

## Stage 10 -- Reporting & Hardening: COMPLETE

Added `reporting/report.py` (`report_to_dict`/`render_markdown_report`,
both derived from the same real result object) and
`scripts/run_regime_health_report.py` (this repo's first CLI entry
point). Added `tests/integration/test_full_system.py` (exact
reproducibility, the real point-in-time no-look-ahead check, and a
minimum-history boundary test) and
`tests/integration/test_cli_workflow.py` (real subprocess CLI test).

65 tests passing (48 unit, 14 integration, 1 CLI subprocess under
`slow`). Full leak-check clean across the whole repo. Pushed.

**All 10 stages of the market-regime-strategy-health-engine build are
now complete.**
