# Data dictionary

Every schema field this engine actually defines, plus the real formula
each build stage computes -- matching the other four engines' own
`docs/data_dictionary.md` convention. Fields reused directly from
upstream engines (e.g. `OptimizationResult`, `BacktestResult`,
`FactorExposureRow`, `FactorStabilityResult`) are not re-documented
here; see their own dictionaries. This engine never re-estimates a
factor exposure, factor covariance, optimization result, portfolio
risk figure, or transaction cost -- it consumes those as real,
already-given upstream inputs.

## `RegimeFeatureRow` (`data/schemas.py`)

- `as_of` -- the date this feature reading is for.
- `realized_volatility` -- trailing per-period (not annualized)
  standard deviation of the real equal-weighted cross-sectional return
  series. `None` when fewer than 2 real trailing returns are
  available, never fabricated as `0.0`.
- `trend_strength` -- the real trailing cumulative return of that same
  series over `trend_window` periods.
- `liquidity_score` -- the real relative change in cross-sectional
  median ADV between the start and end of `liquidity_window`. `None`
  when ADV is missing at either endpoint.
- `avg_pairwise_correlation` -- the real mean off-diagonal entry of the
  per-security return correlation matrix over `correlation_window`.
  `None` below a 2-security cross-section.
- `n_securities` -- the real point-in-time universe size as of this
  date.
- `n_periods_available` -- the real number of market-wide return
  observations actually used -- the figure `classify_regime` checks
  against `min_periods_for_classification`.
- `quality_flags` -- real, specific reasons any feature above is
  `None` (e.g. `"insufficient_return_history"`,
  `"insufficient_cross_section_for_correlation"`,
  `"missing_adv_for_liquidity_feature"`).

## `RegimeClassification` (`data/schemas.py`)

- `label` -- one of `LOW_VOL_TRENDING`, `LOW_VOL_RANGE_BOUND`,
  `HIGH_VOL_TRENDING`, `HIGH_VOL_STRESSED`, `LIQUIDITY_STRESSED`,
  `AMBIGUOUS`, `UNKNOWN`.
- `evidence` -- every real feature value compared against a threshold,
  ALWAYS present (as `NaN` when a feature was unavailable) -- a
  classification can be fully audited even when it lands on
  `AMBIGUOUS`/`UNKNOWN`.
- `triggered_rules` -- the real, human-readable rule name(s) that
  produced this label (e.g. `"liquidity_stress_dominant"`,
  `"high_vol_high_correlation_stress"`); empty for `UNKNOWN`.
- `warnings` -- carries forward any real `RegimeFeatureRow.quality_flags`
  plus this stage's own (e.g. "feature combination matched no real
  rule").

## `StrategyPerformanceSnapshot` (`data/schemas.py`)

- `gross_return`, `net_return`, `turnover`, `transaction_cost` -- read
  directly from `portfolio_construction.backtest.BacktestResult`/
  `OptimizationResult`, never recomputed. `turnover` is attributed to
  the holding period using the SAME real pairing
  `run_backtest` already uses for transaction cost (the rebalance that
  ESTABLISHED a holding period's weights is attributed to that
  period's end date).
- `cumulative_net_return` -- `wealth_index - 1`, where
  `wealth_index = (1 + net_returns).cumprod()` over the real net-return
  series.
- `drawdown` -- `wealth_index / wealth_index.cummax() - 1`, the
  standard real wealth-index drawdown formula.
- `rolling_sharpe` -- `alpha_validation_toolkit.metrics.sharpe_ratio`
  applied to the real trailing `rolling_sharpe_window` of net returns;
  `None` below `min_periods_for_rolling_sharpe`.

## `SignalHealthResult` (`data/schemas.py`)

- `historical_ic_mean`, `recent_ic_mean`, `icir` -- read directly from
  `factor_model.monitoring.compute_factor_stability`'s own real
  `FactorStabilityResult` for the named factor -- the same real,
  point-in-time, cross-sectional lagged-exposure-vs-realized-return IC
  that stage already computes.
- `decay_curve` -- `dict[horizon, float]`: for each real horizon `h`,
  the mean real cross-sectional IC between a security's real lagged
  exposure at `t_prev` and its real realized return from `t_prev` to
  the date `h` real periods later. At `h=1` this is the same real
  quantity `historical_ic_mean` measures; larger `h` answers "how many
  periods can this exposure reading be acted on before it goes stale."
- `exposure_drift` -- `dict[factor_name, float]`: the real |change| in
  the PORTFOLIO's own factor exposure vector between the two most
  recent real snapshots -- a genuinely different real question
  ("is my portfolio's own conviction shifting") from factor-wide
  predictive-power stability.
- `n_periods` -- the real count of non-null IC observations behind
  `historical_ic_mean`.
- `warnings` -- carries forward `FactorStabilityResult`'s own real
  drift/sign-flip warnings unchanged, plus this engine's own (e.g.
  insufficient history for a decay horizon).

## `RegimeConditionedResult` (`data/schemas.py`)

- `regime`, `n_periods`, `mean_net_return`, `volatility_of_net_return`,
  `sharpe` -- real descriptive statistics over exactly the real net
  returns whose matching `RegimeClassification.as_of` carries this
  `regime` label. `sharpe` reuses `alpha_validation_toolkit`'s own
  `sharpe_ratio`; both volatility and Sharpe are withheld (`None`/
  `NaN`) below 2 real observed periods.
- `is_association_not_causation` -- always `True`. A strategy
  performing differently across regimes is a real, observed
  correlation, never proof the regime caused it.

## `DiagnosticFinding` / `Alert` (`data/schemas.py`)

- `DiagnosticFinding.category` -- one of the five real
  `DiagnosticCategory` values: `DATA_QUALITY`,
  `SIGNAL_DETERIORATION`, `FACTOR_EXPOSURE_DRIFT`,
  `MARKET_REGIME_CHANGE`, `EXECUTION_COST_DETERIORATION`.
  `is_causal_claim` is always `False`.
- `Alert.category` -- any of the five above, plus a sixth,
  `STRATEGY_PERFORMANCE` (drawdown/rolling-Sharpe alerts) -- used only
  by `Alert`, never by `DiagnosticFinding`, since it flags the
  strategy's own realized performance directly rather than
  diagnosing a cause. `Alert.severity` is `"warning"` or `"critical"`.

## Stage 2 -- regime features (`regime/features.py`)

`compute_regime_features(market_observations, universe_ids_by_date,
period_dates, as_of, regime_cfg)` requires `as_of` to be the last real
(sorted) element of `period_dates` -- the same PIT discipline every
other engine's own pipeline enforces. Builds the real equal-weighted
cross-sectional return series via `factor_model.regression.
_period_realized_returns` (never re-derived), then:

```
realized_volatility = std(trailing volatility_window of the return series)
trend_strength       = cumulative return over trailing trend_window
avg_pairwise_correlation = mean off-diagonal entry of the per-security
                           return correlation matrix over correlation_window
liquidity_score      = median_ADV(end) / median_ADV(start) - 1
                        over the trailing liquidity_window
```

## Stage 3 -- classification (`regime/classification.py`)

`classify_regime(features, regime_cfg)`: real rule priority (documented
in the module itself, not hidden) -- liquidity stress dominates, then
high-vol-plus-high-correlation stress, then the remaining vol/trend
combinations; anything else (including a missing required feature) is
`AMBIGUOUS`. Below `min_periods_for_classification`, the result is
`UNKNOWN` regardless of the feature values.

## Stage 4 -- strategy performance (`strategy_health/performance.py`)

`build_strategy_performance_snapshots(backtest_result,
optimization_results, portfolio_id, strategy_health_cfg)` -- see the
`StrategyPerformanceSnapshot` formulas above; one real snapshot per
date in `backtest_result.net_returns`.

## Stage 5 -- signal health (`strategy_health/signal_health.py`)

`compute_signal_health(market_observations, fundamental_observations,
universe_snapshot_by_date, exposure_rows, sorted_dates, portfolio_id,
factor_name, as_of, fm_engine_cfg, signal_health_cfg)` -- `sorted_dates`
must already be filtered by the caller to real dates `<= as_of` (the
PIT discipline lives one level up, in the pipeline that assembles this
date range). See the `SignalHealthResult` formulas above for exactly
what is reused vs. newly computed.

## Stage 6 -- regime-conditioned performance (`regime_conditioned/conditioned_performance.py`)

`compute_regime_conditioned_performance(net_returns, classifications,
portfolio_id)` -- groups each real net-return observation by the real
regime label classified for that same date; a date with no matching
classification is excluded, never silently assigned a label it wasn't
given.

## Stage 7 -- diagnostics (`diagnostics/diagnostic_engine.py`)

`run_diagnostics(...)` concatenates five independently-testable real
functions:

- `diagnose_data_quality` -- one finding per real
  `DataHealthReport.warnings` entry.
- `diagnose_signal_deterioration` -- one finding per real
  `SignalHealthResult.warnings` entry (which already carries
  `factor_model.monitoring`'s own drift/sign-flip warnings).
- `diagnose_exposure_drift` -- one finding per real factor whose
  `exposure_drift` exceeds `diagnostics_cfg.exposure_drift_threshold`.
- `diagnose_regime_change` -- fires only when the real label actually
  changed between two consecutive real classifications.
- `diagnose_cost_deterioration` -- fires when the latest real
  `transaction_cost` exceeds its real trailing average by more than
  `diagnostics_cfg.cost_deterioration_pct_threshold`.

## Stage 8 -- alerts (`alerts/alerts.py`)

`generate_alerts(snapshot, signal_health, findings, alerts_cfg)`
applies severity-tiered threshold comparisons to already-computed real
figures -- drawdown and rolling Sharpe (`check_strategy_performance_alerts`),
recent IC level and exposure drift (`check_signal_health_alerts`), and
promoted cost-deterioration findings (`check_cost_alerts`) -- never
re-deriving any underlying real computation.

## Stage 9 -- pipeline (`pipeline/pipeline.py`)

`run_full_regime_health_analysis(...)` runs Stages 1-8 once, in
dependency order, for one real portfolio's real rebalancing history,
requiring `as_of` to be the last (sorted) element of `period_dates` and
at least 2 real `optimization_results` on or before `as_of`. Builds a
real point-in-time universe per date via `factor_model.data.universe.
build_point_in_time_universe`, runs real data-health diagnostics via
`factor_model.diagnostics.data_health.run_data_health_report`, reuses
`portfolio_construction.backtest.run_backtest` directly for the real
backtest, and builds the portfolio's own real factor exposure history
via `factor_model.factors.compute_factor_exposure_table` +
`factor_model.exposures.aggregate_exposure` at each real rebalance
date.

## Stage 10 -- reporting (`reporting/report.py`)

`report_to_dict(result)` / `render_markdown_report(result)` both
derive from the same real `RegimeHealthAnalysisResult` -- never
recomputed separately, so they cannot silently disagree.
