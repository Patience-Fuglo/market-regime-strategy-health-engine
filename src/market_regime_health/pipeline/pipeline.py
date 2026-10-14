"""Stage 9 -- the end-to-end orchestration: runs Stages 1-8 once, in
dependency order, for one real portfolio's real rebalancing history.
Every step reuses the exact function that stage (or an upstream
engine) already built and tested -- nothing here re-derives any real
computation. Depends on ``factor-model-engine`` (universe, factor
exposure, data health) and ``portfolio-construction-engine``
(``OptimizationResult``/``run_backtest``) for everything this engine
itself does not define.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd
from factor_model.config import EngineConfig as FactorModelConfig
from factor_model.data.schemas import ExposureSubject, FactorExposureRow, FundamentalObservation, MarketObservation, PortfolioHolding
from factor_model.data.universe import build_point_in_time_universe
from factor_model.diagnostics.data_health import DataHealthReport, run_data_health_report
from factor_model.exposures import aggregate_exposure
from factor_model.factors import compute_factor_exposure_table
from portfolio_construction.backtest import run_backtest
from portfolio_construction.data import OptimizationResult

from ..alerts import generate_alerts
from ..config import EngineConfig
from ..data.schemas import Alert, DiagnosticFinding, RegimeClassification, RegimeConditionedResult, SignalHealthResult, StrategyPerformanceSnapshot
from ..diagnostics import run_diagnostics
from ..regime import classify_regime, compute_regime_features
from ..regime_conditioned import compute_regime_conditioned_performance
from ..strategy_health import build_strategy_performance_snapshots, compute_signal_health


@dataclass
class RegimeHealthAnalysisResult:
    portfolio_id: str
    as_of: pd.Timestamp
    signal_factor_name: str
    regime_classification: RegimeClassification
    strategy_performance: StrategyPerformanceSnapshot
    signal_health: SignalHealthResult
    data_health: DataHealthReport
    findings: list[DiagnosticFinding]
    alerts: list[Alert]
    regime_conditioned: list[RegimeConditionedResult] = field(default_factory=list)
    classification_history: list[RegimeClassification] = field(default_factory=list)
    performance_history: list[StrategyPerformanceSnapshot] = field(default_factory=list)


def run_full_regime_health_analysis(
    market_observations: Sequence[MarketObservation],
    fundamental_observations: Sequence[FundamentalObservation],
    optimization_results: Sequence[OptimizationResult],
    period_dates: Sequence[pd.Timestamp],
    as_of: pd.Timestamp,
    signal_factor_name: str,
    fm_engine_cfg: FactorModelConfig,
    engine_cfg: EngineConfig,
    portfolio_id: str = "portfolio",
) -> RegimeHealthAnalysisResult:
    """``period_dates`` must have ``as_of`` as its last (sorted)
    element, and ``optimization_results`` must include at least 2 real
    rebalance dates on or before ``as_of`` -- the same real PIT
    discipline every other engine's own pipeline enforces.
    """
    sorted_dates = sorted(period_dates)
    if not sorted_dates or sorted_dates[-1] != as_of:
        raise ValueError("as_of must be the last real date in period_dates")

    own_results = sorted((r for r in optimization_results if r.as_of <= as_of), key=lambda r: r.as_of)
    if len(own_results) < 2:
        raise ValueError("need at least 2 real optimization results on or before as_of")

    universe_snapshot_by_date = {
        d: build_point_in_time_universe(market_observations, d, fm_engine_cfg.universe, fm_engine_cfg.data) for d in sorted_dates
    }
    universe_ids_by_date = {d: snap.included_ids() for d, snap in universe_snapshot_by_date.items()}

    data_health_report = run_data_health_report(
        market_observations, fundamental_observations, as_of, fm_engine_cfg.data, fm_engine_cfg.diagnostics
    )

    classification_history: list[RegimeClassification] = []
    for result in own_results:
        d = result.as_of
        trailing_dates = [dt for dt in sorted_dates if dt <= d]
        if len(trailing_dates) < 2:
            continue
        features = compute_regime_features(market_observations, universe_ids_by_date, trailing_dates, d, engine_cfg.regime)
        classification_history.append(classify_regime(features, engine_cfg.regime))
    if not classification_history:
        raise ValueError("no real regime classification could be produced -- check period_dates/optimization_results overlap")
    classification = next(c for c in reversed(classification_history) if c.as_of == as_of)
    prior_classification = classification_history[-2] if len(classification_history) >= 2 and classification_history[-1].as_of == as_of else None

    backtest_result = run_backtest(own_results, market_observations)
    performance_history = build_strategy_performance_snapshots(backtest_result, own_results, portfolio_id, engine_cfg.strategy_health)
    snapshot = next((s for s in reversed(performance_history) if s.as_of == as_of), None)
    if snapshot is None:
        raise ValueError(f"no real strategy performance snapshot found for as_of={as_of}")

    exposure_rows: list[FactorExposureRow] = []
    for result in own_results:
        d = result.as_of
        stock_exposure_rows, _ = compute_factor_exposure_table(
            market_observations, fundamental_observations, d, universe_snapshot_by_date[d], fm_engine_cfg
        )
        holdings = [
            PortfolioHolding(portfolio_id=portfolio_id, security_id=sid, as_of=d, weight=w, known_as_of=d)
            for sid, w in result.target_weights.items()
        ]
        exposure_rows.extend(aggregate_exposure(stock_exposure_rows, holdings, ExposureSubject.PORTFOLIO, portfolio_id))

    dates_through_as_of = [d for d in sorted_dates if d <= as_of]
    signal_health = compute_signal_health(
        market_observations, fundamental_observations, universe_snapshot_by_date, exposure_rows, dates_through_as_of,
        portfolio_id, signal_factor_name, as_of, fm_engine_cfg, engine_cfg.signal_health,
    )

    regime_conditioned = compute_regime_conditioned_performance(backtest_result.net_returns, classification_history, portfolio_id)

    findings = run_diagnostics(
        as_of, data_health_report, signal_health, classification, prior_classification,
        performance_history, engine_cfg.diagnostics,
    )
    alerts = generate_alerts(snapshot, signal_health, findings, engine_cfg.alerts)

    return RegimeHealthAnalysisResult(
        portfolio_id=portfolio_id,
        as_of=as_of,
        signal_factor_name=signal_factor_name,
        regime_classification=classification,
        strategy_performance=snapshot,
        signal_health=signal_health,
        data_health=data_health_report,
        findings=findings,
        alerts=alerts,
        regime_conditioned=regime_conditioned,
        classification_history=classification_history,
        performance_history=performance_history,
    )
