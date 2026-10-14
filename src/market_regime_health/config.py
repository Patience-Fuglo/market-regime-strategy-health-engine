"""Engine configuration: every ambiguous research decision lives here,
in YAML, with a recommended default and reasoning in
``docs/research_assumptions.md`` -- never hardcoded in source. Same
plain-``@dataclass`` house style as the other four engines in this
build.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass(frozen=True)
class RunConfig:
    output_dir: str = "outputs"
    random_seed: int = 1337


@dataclass(frozen=True)
class RegimeConfig:
    # Every window below is in real "periods," matching whatever
    # periodicity the underlying market observations use (monthly for
    # this engine's synthetic demonstrations, consistent with the other
    # four engines in this build -- see portfolio-risk-engine's own
    # VaRConfig.horizon_periods for the same convention).

    # RESEARCH ASSUMPTION -- trailing periods used for realized volatility
    volatility_window: int = 12
    # RESEARCH ASSUMPTION -- trailing periods used for trend strength
    trend_window: int = 12
    # RESEARCH ASSUMPTION -- trailing periods used for average pairwise
    # cross-sectional correlation
    correlation_window: int = 12
    # RESEARCH ASSUMPTION -- trailing periods used for the liquidity
    # feature (relative change in cross-sectional median ADV)
    liquidity_window: int = 6
    # RESEARCH ASSUMPTION -- below this many real trailing periods of
    # history, classification returns UNKNOWN rather than a guess
    min_periods_for_classification: int = 12
    # RESEARCH ASSUMPTION -- trailing per-period (NOT annualized)
    # standard deviation of market-wide returns above which the
    # volatility feature is classified "high"
    high_vol_threshold: float = 0.06
    # RESEARCH ASSUMPTION -- trailing per-period standard deviation
    # below which the volatility feature is classified "low" (strictly
    # below high_vol_threshold; the band between is "moderate")
    low_vol_threshold: float = 0.03
    # RESEARCH ASSUMPTION -- absolute trailing cumulative return over
    # trend_window above which the market is classified "trending"
    # rather than "range-bound"
    trend_threshold: float = 0.05
    # RESEARCH ASSUMPTION -- average pairwise cross-sectional correlation
    # above which correlation is classified "high" (a real stress
    # indicator -- correlations rise when diversification breaks down)
    high_correlation_threshold: float = 0.45
    # RESEARCH ASSUMPTION -- relative change in the liquidity score over
    # liquidity_window below which liquidity is classified "stressed"
    liquidity_stress_threshold: float = -0.20

    def __post_init__(self) -> None:
        if self.volatility_window < 2 or self.trend_window < 2 or self.correlation_window < 2 or self.liquidity_window < 2:
            raise ValueError("all regime feature windows must be >= 2")
        if self.min_periods_for_classification < 2:
            raise ValueError("min_periods_for_classification must be >= 2")
        if self.low_vol_threshold >= self.high_vol_threshold:
            raise ValueError("low_vol_threshold must be strictly below high_vol_threshold")
        if self.trend_threshold < 0:
            raise ValueError("trend_threshold must be non-negative (compared against an absolute value)")
        if not 0 < self.high_correlation_threshold <= 1:
            raise ValueError("high_correlation_threshold must be in (0, 1]")


@dataclass(frozen=True)
class StrategyHealthConfig:
    # RESEARCH ASSUMPTION -- trailing periods for rolling Sharpe
    rolling_sharpe_window: int = 12
    # RESEARCH ASSUMPTION -- below this many real trailing periods,
    # rolling Sharpe is not computed (reported as None, never a noisy guess)
    min_periods_for_rolling_sharpe: int = 6

    def __post_init__(self) -> None:
        if self.rolling_sharpe_window < 2:
            raise ValueError("rolling_sharpe_window must be >= 2")
        if self.min_periods_for_rolling_sharpe < 2:
            raise ValueError("min_periods_for_rolling_sharpe must be >= 2")


@dataclass(frozen=True)
class SignalHealthConfig:
    # RESEARCH ASSUMPTION -- trailing periods for rolling IC
    rolling_ic_window: int = 12
    # RESEARCH ASSUMPTION -- below this many real trailing periods,
    # rolling IC is not computed for that window (NaN, never a noisy guess)
    rolling_ic_min_periods: int = 6
    # RESEARCH ASSUMPTION -- forward horizons (in real periods) checked
    # by the decay curve
    decay_horizons: tuple[int, ...] = (1, 3, 6, 12)
    # RESEARCH ASSUMPTION -- below this many real paired (signal, forward
    # return) observations, IC/ICIR are still computed but flagged
    # unreliable, never withheld
    min_observations_for_reliable_ic: int = 24

    def __post_init__(self) -> None:
        if self.rolling_ic_window < 2:
            raise ValueError("rolling_ic_window must be >= 2")
        if self.rolling_ic_min_periods < 2:
            raise ValueError("rolling_ic_min_periods must be >= 2")
        if not self.decay_horizons or any(h <= 0 for h in self.decay_horizons):
            raise ValueError("decay_horizons must be non-empty and all positive")
        if self.min_observations_for_reliable_ic < 2:
            raise ValueError("min_observations_for_reliable_ic must be >= 2")


@dataclass(frozen=True)
class DiagnosticsConfig:
    # Real signal-deterioration diagnostics are surfaced directly from
    # factor-model-engine's own MonitoringConfig (drift_recent_window/
    # drift_decay_threshold, applied inside compute_factor_stability) --
    # no separate, parallel IC threshold is defined here, to avoid two
    # independent real checks silently disagreeing on the same question.

    # RESEARCH ASSUMPTION -- the per-factor real change in a portfolio's
    # factor exposure between two consecutive real snapshots above which
    # real exposure drift is flagged
    exposure_drift_threshold: float = 0.15
    # RESEARCH ASSUMPTION -- the real relative increase in transaction
    # cost (current vs. trailing average) above which real execution-cost
    # deterioration is flagged
    cost_deterioration_pct_threshold: float = 0.25

    def __post_init__(self) -> None:
        if self.exposure_drift_threshold < 0:
            raise ValueError("exposure_drift_threshold must be non-negative")
        if self.cost_deterioration_pct_threshold < 0:
            raise ValueError("cost_deterioration_pct_threshold must be non-negative")


@dataclass(frozen=True)
class AlertsConfig:
    # RESEARCH ASSUMPTION -- real drawdown fractions above which a
    # drawdown alert fires at each severity
    drawdown_warning: float = 0.10
    drawdown_critical: float = 0.20
    # RESEARCH ASSUMPTION -- real rolling Sharpe levels below which a
    # strategy-health alert fires at each severity
    rolling_sharpe_warning: float = 0.0
    rolling_sharpe_critical: float = -0.5
    # RESEARCH ASSUMPTION -- real recent-IC levels (an ABSOLUTE level
    # check, distinct from factor-model-engine's own RELATIVE
    # drift-vs-history check already surfaced via diagnose_signal_
    # deterioration) below which a signal-health alert fires at each
    # severity
    signal_ic_warning: float = 0.01
    signal_ic_critical: float = 0.0
    # RESEARCH ASSUMPTION -- real exposure-drift magnitude above which an
    # alert fires at each severity
    exposure_drift_warning: float = 0.15
    exposure_drift_critical: float = 0.30
    # RESEARCH ASSUMPTION -- real relative cost increase above which an
    # alert fires at each severity
    cost_deterioration_warning_pct: float = 0.25
    cost_deterioration_critical_pct: float = 0.50

    def __post_init__(self) -> None:
        if self.drawdown_warning > self.drawdown_critical:
            raise ValueError("drawdown_warning cannot exceed drawdown_critical")
        if self.rolling_sharpe_warning < self.rolling_sharpe_critical:
            raise ValueError("rolling_sharpe_warning cannot be below rolling_sharpe_critical")
        if self.signal_ic_warning < self.signal_ic_critical:
            raise ValueError("signal_ic_warning cannot be below signal_ic_critical")
        if self.exposure_drift_warning > self.exposure_drift_critical:
            raise ValueError("exposure_drift_warning cannot exceed exposure_drift_critical")
        if self.cost_deterioration_warning_pct > self.cost_deterioration_critical_pct:
            raise ValueError("cost_deterioration_warning_pct cannot exceed cost_deterioration_critical_pct")


@dataclass(frozen=True)
class EngineConfig:
    run: RunConfig = field(default_factory=RunConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    strategy_health: StrategyHealthConfig = field(default_factory=StrategyHealthConfig)
    signal_health: SignalHealthConfig = field(default_factory=SignalHealthConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)


def load_config(path: str) -> EngineConfig:
    """Load a real YAML config file into a validated EngineConfig.
    Missing sections fall back to each dataclass's own documented
    default -- never a silent empty/zero value."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    run = RunConfig(**raw.get("run", {}))
    regime_raw = dict(raw.get("regime", {}))
    if "decay_horizons" in regime_raw:
        regime_raw["decay_horizons"] = tuple(regime_raw["decay_horizons"])
    regime = RegimeConfig(**regime_raw)
    strategy_health = StrategyHealthConfig(**raw.get("strategy_health", {}))
    signal_health_raw = dict(raw.get("signal_health", {}))
    if "decay_horizons" in signal_health_raw:
        signal_health_raw["decay_horizons"] = tuple(signal_health_raw["decay_horizons"])
    signal_health = SignalHealthConfig(**signal_health_raw)
    diagnostics = DiagnosticsConfig(**raw.get("diagnostics", {}))
    alerts = AlertsConfig(**raw.get("alerts", {}))

    return EngineConfig(
        run=run, regime=regime, strategy_health=strategy_health,
        signal_health=signal_health, diagnostics=diagnostics, alerts=alerts,
    )
