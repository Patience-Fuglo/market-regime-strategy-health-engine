"""Core data schemas for the market regime and strategy health engine.

This engine never re-estimates a factor exposure, factor covariance,
optimization result, portfolio risk figure, or transaction cost -- it
consumes ``factor-model-engine``'s, ``portfolio-construction-engine``'s,
``portfolio-risk-engine``'s, and ``transaction-cost-engine``'s own real
outputs directly. Every schema here answers one of: "what market regime
is present," "is the strategy behaving as expected," "is the signal
still healthy," or "what, specifically, explains any deterioration" --
never "which portfolio should we hold" or "how risky is this portfolio"
(those are earlier engines' jobs).

Six structurally separate concepts live here, matching the other four
engines' own convention of never merging distinct real questions into
one table:

1. ``RegimeFeatureRow`` -- real, point-in-time market-wide feature
   readings (volatility, trend, liquidity, correlation, stress).
2. ``RegimeClassification`` -- a transparent real LABEL derived from
   those features, with the exact evidence and rules that produced it.
3. ``StrategyPerformanceSnapshot`` -- real strategy performance as of
   one date (gross/net return, drawdown, rolling Sharpe, turnover, cost).
4. ``SignalHealthResult`` -- real signal quality (IC, decay, stability,
   exposure drift) as of one date.
5. ``RegimeConditionedResult`` -- real strategy performance statistics,
   conditioned on (grouped by) regime label -- an ASSOCIATION, never
   presented as causation.
6. ``DiagnosticFinding`` / ``Alert`` -- a real, evidenced explanation of
   what changed, and (when a configured threshold is crossed) a
   severity-tagged alert built from that same evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class RegimeLabel(str, Enum):
    """A transparent, rule-based regime label. ``UNKNOWN`` means
    insufficient real history to classify at all; ``AMBIGUOUS`` means
    there WAS enough history, but the real feature readings did not
    cleanly match any one real rule (e.g. high volatility with low
    correlation, which this engine's rules deliberately do not resolve
    into a single label rather than guessing)."""

    LOW_VOL_TRENDING = "low_vol_trending"
    LOW_VOL_RANGE_BOUND = "low_vol_range_bound"
    HIGH_VOL_TRENDING = "high_vol_trending"
    HIGH_VOL_STRESSED = "high_vol_stressed"
    LIQUIDITY_STRESSED = "liquidity_stressed"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


class DiagnosticCategory(str, Enum):
    """The five real deterioration SOURCES this engine's diagnostic
    framework distinguishes between -- per the engine's own core
    purpose, these are kept mutually distinct rather than collapsed
    into one generic "something is wrong" signal. ``STRATEGY_
    PERFORMANCE`` is a sixth, real category used only by ``Alert``
    (never by ``DiagnosticFinding``): it flags the strategy's own
    realized performance (drawdown, rolling Sharpe) crossing a
    threshold directly -- "is the strategy behaving as expected," a
    real question distinct from "why," which the other five answer.
    """

    DATA_QUALITY = "data_quality"
    SIGNAL_DETERIORATION = "signal_deterioration"
    FACTOR_EXPOSURE_DRIFT = "factor_exposure_drift"
    MARKET_REGIME_CHANGE = "market_regime_change"
    EXECUTION_COST_DETERIORATION = "execution_cost_deterioration"
    STRATEGY_PERFORMANCE = "strategy_performance"


@dataclass(frozen=True)
class RegimeFeatureRow:
    """Real, point-in-time market-wide feature readings as of one date.
    Each feature is ``None`` (never a fabricated 0.0) when the real
    trailing window does not yet have enough history -- see
    ``n_periods_available`` for exactly how much history was actually
    used.
    """

    as_of: pd.Timestamp
    realized_volatility: float | None
    trend_strength: float | None
    liquidity_score: float | None
    avg_pairwise_correlation: float | None
    n_securities: int
    n_periods_available: int
    quality_flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RegimeClassification:
    """A real, transparent regime label, together with the exact
    evidence (feature values) and the human-readable rule names that
    actually fired -- never a black-box label. ``evidence`` always
    carries every real feature value compared against a threshold,
    whether or not that feature's rule fired, so a researcher can audit
    a classification even when it lands on ``UNKNOWN``/``AMBIGUOUS``.
    """

    as_of: pd.Timestamp
    label: RegimeLabel
    evidence: dict[str, float]
    triggered_rules: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StrategyPerformanceSnapshot:
    """Real strategy performance as of one date. ``gross_return``/
    ``net_return``/``turnover``/``transaction_cost`` are never
    recomputed here -- they are read directly from
    ``portfolio_construction.backtest.BacktestResult``/
    ``OptimizationResult``, the real upstream source of truth for a
    strategy's own realized performance. ``drawdown`` and
    ``rolling_sharpe`` are this engine's own real additions, computed
    from the real net-return series those upstream results provide.
    """

    as_of: pd.Timestamp
    portfolio_id: str
    gross_return: float
    net_return: float
    cumulative_net_return: float
    drawdown: float
    rolling_sharpe: float | None
    turnover: float
    transaction_cost: float
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SignalHealthResult:
    """Real signal quality for one named factor, as of one date.
    ``historical_ic_mean``/``recent_ic_mean``/``icir`` are never
    reimplemented here -- they are read directly from
    ``factor_model.monitoring.compute_factor_stability``'s own real
    ``FactorStabilityResult`` (the same real, point-in-time,
    cross-sectional lagged-exposure-vs-realized-return IC that stage
    already computes and already checks for drift/sign-flip -- see
    ``warnings``, carried through unchanged). This engine adds two real
    capabilities ``factor_model.monitoring`` does not provide:

    - ``decay_curve``: the SAME real cross-sectional IC computation,
      generalized across multiple forward horizons (``factor_model``'s
      own stability check only ever looks one period ahead).
    - ``exposure_drift``: the change in a specific PORTFOLIO's own real
      factor exposure vector (``factor_model.data.schemas.
      FactorExposureRow``, subject_type PORTFOLIO) between two
      consecutive real snapshots -- a genuinely different real
      question ("is my portfolio's own conviction shifting") from
      factor-wide predictive-power stability.
    """

    as_of: pd.Timestamp
    signal_name: str
    historical_ic_mean: float
    recent_ic_mean: float
    icir: float
    decay_curve: dict[int, float] = field(default_factory=dict)
    exposure_drift: dict[str, float] = field(default_factory=dict)
    n_periods: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RegimeConditionedResult:
    """Real descriptive strategy-performance statistics, computed only
    over the real historical periods labeled with ``regime`` --
    ``is_association_not_causation`` is always ``True`` and exists
    specifically so no downstream consumer (a report, a dashboard, a
    future engine) can accidentally present this as a causal claim: a
    strategy performing poorly during a given regime is a real
    correlation, never proof that the regime caused the poor
    performance (the same conflation `factor-model-engine`'s own
    attribution schema was designed to prevent between exposure and
    P&L contribution).
    """

    portfolio_id: str
    regime: RegimeLabel
    n_periods: int
    mean_net_return: float
    volatility_of_net_return: float
    sharpe: float | None
    is_association_not_causation: bool = True
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DiagnosticFinding:
    """A real, evidenced explanation of what changed, in one of the five
    real ``DiagnosticCategory`` buckets. ``is_causal_claim`` is always
    ``False`` -- a finding states an observed association between a
    real metric and a real threshold, never that one caused the other
    (a ``MARKET_REGIME_CHANGE`` finding, in particular, must never be
    read as "the regime caused the strategy's poor performance").
    """

    as_of: pd.Timestamp
    category: DiagnosticCategory
    description: str
    evidence: dict[str, float]
    is_causal_claim: bool = False


@dataclass(frozen=True)
class Alert:
    """One real, severity-tagged alert, built directly from a
    ``DiagnosticFinding``'s own real evidence once it crosses a
    configured threshold (see ``alerts.alerts``) -- never a
    freestanding judgment independent of that evidence.
    """

    as_of: pd.Timestamp
    category: DiagnosticCategory
    severity: str
    message: str
    evidence: dict[str, float]

    def __post_init__(self) -> None:
        if self.severity not in ("warning", "critical"):
            raise ValueError(f"unknown severity {self.severity!r}")
