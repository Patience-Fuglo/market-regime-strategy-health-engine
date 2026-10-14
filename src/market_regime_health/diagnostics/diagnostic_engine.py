"""Stage 7 -- the real diagnostic framework: distinguishes data
quality, signal deterioration, factor exposure drift, market regime
change, and execution-cost deterioration as five SEPARATE real
categories, each backed by its own real evidence -- never collapsed
into one generic "something is wrong" signal. Every finding here
states an observed association with a real threshold, never a causal
claim (``DiagnosticFinding.is_causal_claim`` is always ``False``) -- a
``MARKET_REGIME_CHANGE`` finding, in particular, must never be read as
"the regime caused the strategy's poor performance," only that both
happened at the same time.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from factor_model.diagnostics.data_health import DataHealthReport

from ..config import DiagnosticsConfig
from ..data.schemas import (
    DiagnosticCategory,
    DiagnosticFinding,
    RegimeClassification,
    SignalHealthResult,
    StrategyPerformanceSnapshot,
)


def diagnose_data_quality(as_of: pd.Timestamp, data_health_report: DataHealthReport) -> list[DiagnosticFinding]:
    """One real finding per real warning ``factor-model-engine``'s own
    ``run_data_health_report`` already produced -- never re-derived.
    """
    return [
        DiagnosticFinding(
            as_of=as_of,
            category=DiagnosticCategory.DATA_QUALITY,
            description=warning,
            evidence={"n_securities_observed": float(data_health_report.n_securities_observed)},
        )
        for warning in data_health_report.warnings
    ]


def diagnose_signal_deterioration(signal_health: SignalHealthResult) -> list[DiagnosticFinding]:
    """One real finding per real warning
    ``factor_model.monitoring.compute_factor_stability`` already
    produced (sign-flip or decay vs. the factor's own history) --
    reused directly via ``signal_health.warnings``, never re-derived
    with a second, parallel threshold.
    """
    return [
        DiagnosticFinding(
            as_of=signal_health.as_of,
            category=DiagnosticCategory.SIGNAL_DETERIORATION,
            description=warning,
            evidence={
                "historical_ic_mean": signal_health.historical_ic_mean,
                "recent_ic_mean": signal_health.recent_ic_mean,
            },
        )
        for warning in signal_health.warnings
    ]


def diagnose_exposure_drift(signal_health: SignalHealthResult, diagnostics_cfg: DiagnosticsConfig) -> list[DiagnosticFinding]:
    return [
        DiagnosticFinding(
            as_of=signal_health.as_of,
            category=DiagnosticCategory.FACTOR_EXPOSURE_DRIFT,
            description=(
                f"{factor_name!r} exposure drifted by {drift:.4f}, above the configured threshold "
                f"({diagnostics_cfg.exposure_drift_threshold:.4f})"
            ),
            evidence={"drift": drift, "threshold": diagnostics_cfg.exposure_drift_threshold},
        )
        for factor_name, drift in signal_health.exposure_drift.items()
        if drift > diagnostics_cfg.exposure_drift_threshold
    ]


def diagnose_regime_change(
    classification: RegimeClassification, prior_classification: RegimeClassification | None
) -> list[DiagnosticFinding]:
    if prior_classification is None or classification.label == prior_classification.label:
        return []
    return [
        DiagnosticFinding(
            as_of=classification.as_of,
            category=DiagnosticCategory.MARKET_REGIME_CHANGE,
            description=(
                f"regime classification changed from {prior_classification.label.value!r} to "
                f"{classification.label.value!r} -- an observed association with any concurrent "
                "performance change, not a causal claim"
            ),
            evidence=dict(classification.evidence),
        )
    ]


def diagnose_cost_deterioration(
    snapshots: Sequence[StrategyPerformanceSnapshot], diagnostics_cfg: DiagnosticsConfig
) -> list[DiagnosticFinding]:
    sorted_snapshots = sorted(snapshots, key=lambda s: s.as_of)
    if len(sorted_snapshots) < 2:
        return []
    latest = sorted_snapshots[-1]
    prior_costs = [s.transaction_cost for s in sorted_snapshots[:-1] if pd.notna(s.transaction_cost)]
    if not prior_costs:
        return []
    trailing_average = sum(prior_costs) / len(prior_costs)
    if trailing_average <= 0 or pd.isna(latest.transaction_cost):
        return []
    relative_increase = latest.transaction_cost / trailing_average - 1.0
    if relative_increase <= diagnostics_cfg.cost_deterioration_pct_threshold:
        return []
    return [
        DiagnosticFinding(
            as_of=latest.as_of,
            category=DiagnosticCategory.EXECUTION_COST_DETERIORATION,
            description=(
                f"transaction cost ({latest.transaction_cost:.4%}) is {relative_increase:.1%} above its trailing "
                f"average ({trailing_average:.4%}), above the configured threshold "
                f"({diagnostics_cfg.cost_deterioration_pct_threshold:.1%})"
            ),
            evidence={
                "latest_transaction_cost": latest.transaction_cost,
                "trailing_average_transaction_cost": trailing_average,
                "relative_increase": relative_increase,
            },
        )
    ]


def run_diagnostics(
    as_of: pd.Timestamp,
    data_health_report: DataHealthReport,
    signal_health: SignalHealthResult,
    classification: RegimeClassification,
    prior_classification: RegimeClassification | None,
    snapshots: Sequence[StrategyPerformanceSnapshot],
    diagnostics_cfg: DiagnosticsConfig,
) -> list[DiagnosticFinding]:
    """Runs every real diagnostic category and concatenates the real
    findings -- each category's own function is independently testable
    and never shares state with the others.
    """
    findings: list[DiagnosticFinding] = []
    findings += diagnose_data_quality(as_of, data_health_report)
    findings += diagnose_signal_deterioration(signal_health)
    findings += diagnose_exposure_drift(signal_health, diagnostics_cfg)
    findings += diagnose_regime_change(classification, prior_classification)
    findings += diagnose_cost_deterioration(snapshots, diagnostics_cfg)
    return findings
