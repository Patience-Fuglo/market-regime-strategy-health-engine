# Research assumptions

Every field below is a real modeling judgment call, not a safe
engineering default, and needs the researcher's explicit sign-off
before being relied on for any real analysis. Each lives in
`configs/default.yaml`, tagged `# RESEARCH ASSUMPTION` at the point of
definition, and in `src/market_regime_health/config.py`'s own inline
comments -- this document consolidates both into one reference,
matching the other four engines' own convention.

All windows below are in real "periods," matching whatever periodicity
the underlying market observations use (monthly for this engine's
synthetic demonstrations, consistent with every other engine in this
build).

## Regime (`regime:` in `configs/default.yaml`)

**`volatility_window: 12`**, **`trend_window: 12`**,
**`correlation_window: 12`**, **`liquidity_window: 6`**
The trailing lengths used for each real market-wide feature
(`regime/features.py`). 12 real periods (a year, at monthly
cadence) is a conventional lookback for a volatility/trend/correlation
regime read; liquidity is given a shorter 6-period window since ADV
shifts are expected to be felt faster than a volatility or correlation
regime shift.

**`min_periods_for_classification: 12`**
Below this many real trailing periods, `classify_regime` returns
`UNKNOWN` rather than a guess built on too little history.

**`high_vol_threshold: 0.06`**, **`low_vol_threshold: 0.03`**
The trailing per-period (NOT annualized) standard deviation of the
real market-wide return series above/below which volatility is
classified "high"/"low." Calibrated for monthly equity-like returns
(roughly 10%/21% annualized at the low/high boundary via
`sqrt(12)` scaling) -- a different real data periodicity should
rescale these.

**`trend_threshold: 0.05`**
The absolute trailing cumulative return above which the market is
classified "trending" rather than "range-bound." A round, moderate
5% over a trailing year.

**`high_correlation_threshold: 0.45`**
The average pairwise cross-sectional correlation above which
correlation is classified "high" -- a real stress indicator, since
correlations across securities tend to rise together when
diversification breaks down.

**`liquidity_stress_threshold: -0.20`**
The relative change in the liquidity feature over `liquidity_window`
below which liquidity is classified "stressed" -- a 20% real drop in
cross-sectional median ADV.

## Strategy health (`strategy_health:`)

**`rolling_sharpe_window: 12`**
The trailing real periods used for rolling Sharpe
(`strategy_health/performance.py`), reusing
`alpha-validation-toolkit`'s own real `sharpe_ratio` per window.

**`min_periods_for_rolling_sharpe: 6`**
Below this many real trailing periods, rolling Sharpe is withheld
(`None`), never computed on too little history.

## Signal health (`signal_health:`)

**`rolling_ic_window: 12`**, **`rolling_ic_min_periods: 6`**
Carried over from this engine's own schema for consistency, though the
PRIMARY real IC/ICIR figures are now read directly from
`factor-model-engine`'s own `MonitoringConfig` (`drift_recent_window`/
`drift_decay_threshold`) rather than independently windowed here --
see the Stage 5 note in `docs/STATUS.md`.

**`decay_horizons: (1, 3, 6, 12)`**
The real forward horizons checked by the multi-horizon decay curve --
one period ahead through a full real year, at monthly cadence.

**`min_observations_for_reliable_ic: 24`**
Below this many real periods of IC history, the signal-health result
still reports its real numbers (never withheld) but is flagged
unreliable.

## Diagnostics (`diagnostics:`)

Real signal-deterioration diagnostics are surfaced directly from
`factor-model-engine`'s own `MonitoringConfig` (applied inside
`compute_factor_stability`) -- no separate, parallel IC threshold is
defined here, to avoid two independent real checks silently
disagreeing on the same real question.

**`exposure_drift_threshold: 0.15`**
The per-factor real change in a portfolio's own factor exposure
between two consecutive real snapshots above which real exposure
drift is flagged as a diagnostic finding.

**`cost_deterioration_pct_threshold: 0.25`**
The real relative increase in transaction cost (current vs. trailing
average) above which real execution-cost deterioration is flagged.

## Alerts (`alerts:`)

**`drawdown_warning: 0.10`**, **`drawdown_critical: 0.20`**
Real drawdown fractions above which a strategy-performance alert fires
at each severity.

**`rolling_sharpe_warning: 0.0`**, **`rolling_sharpe_critical: -0.5`**
Real rolling Sharpe levels below which a strategy-performance alert
fires at each severity.

**`signal_ic_warning: 0.01`**, **`signal_ic_critical: 0.0`**
Real `recent_ic_mean` levels below which a signal-health alert fires.
This is deliberately an ABSOLUTE level check, distinct from
`factor-model-engine`'s own RELATIVE drift-vs-history check already
surfaced via `diagnose_signal_deterioration` -- a signal can be
absolutely weak without having recently drifted, and vice versa; both
are real, non-redundant questions.

**`exposure_drift_warning: 0.15`**, **`exposure_drift_critical: 0.30`**
Real per-factor exposure-drift magnitudes above which an alert fires
at each severity.

**`cost_deterioration_warning_pct: 0.25`**, **`cost_deterioration_critical_pct: 0.50`**
Real relative transaction-cost increases above which an alert fires at
each severity.

## Not a research assumption (real engineering defaults)

`run.output_dir`, `run.random_seed` are safe engineering defaults, not
modeling choices, and are not listed here. `volatility.py`-style
numerical-noise floors do not apply to this engine (it performs no
near-zero-magnitude BLAS quadratic form of its own).

## Reused real inputs, not re-estimated here

This engine makes no independent research assumptions about factor
construction, factor covariance, optimization objectives,
transaction-cost models, or VaR/ES -- those real judgment calls live in
`factor-model-engine`'s, `portfolio-construction-engine`'s,
`portfolio-risk-engine`'s, and `transaction-cost-engine`'s own
`docs/research_assumptions.md` files and are consumed here as given
(see `docs/data_dictionary.md` for exactly which real objects cross
that boundary).
