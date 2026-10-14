"""Stage 8 -- real, severity-tiered alerts, built directly from
already-computed real figures (``StrategyPerformanceSnapshot``,
``SignalHealthResult``, ``DiagnosticFinding``) -- mirrors
``portfolio-risk-engine``'s own ``limits.check_limits`` convention:
this stage applies NEW, configured threshold comparisons, it never
re-derives any underlying real computation.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from ..config import AlertsConfig
from ..data.schemas import Alert, DiagnosticCategory, DiagnosticFinding, SignalHealthResult, StrategyPerformanceSnapshot


def check_strategy_performance_alerts(snapshot: StrategyPerformanceSnapshot, alerts_cfg: AlertsConfig) -> list[Alert]:
    alerts: list[Alert] = []

    if snapshot.drawdown <= -alerts_cfg.drawdown_critical:
        severity = "critical"
    elif snapshot.drawdown <= -alerts_cfg.drawdown_warning:
        severity = "warning"
    else:
        severity = None
    if severity is not None:
        alerts.append(
            Alert(
                as_of=snapshot.as_of,
                category=DiagnosticCategory.STRATEGY_PERFORMANCE,
                severity=severity,
                message=f"drawdown ({snapshot.drawdown:.2%}) breached the {severity} threshold",
                evidence={"drawdown": snapshot.drawdown},
            )
        )

    if snapshot.rolling_sharpe is not None:
        if snapshot.rolling_sharpe <= alerts_cfg.rolling_sharpe_critical:
            severity = "critical"
        elif snapshot.rolling_sharpe <= alerts_cfg.rolling_sharpe_warning:
            severity = "warning"
        else:
            severity = None
        if severity is not None:
            alerts.append(
                Alert(
                    as_of=snapshot.as_of,
                    category=DiagnosticCategory.STRATEGY_PERFORMANCE,
                    severity=severity,
                    message=f"rolling Sharpe ({snapshot.rolling_sharpe:.2f}) breached the {severity} threshold",
                    evidence={"rolling_sharpe": snapshot.rolling_sharpe},
                )
            )

    return alerts


def check_signal_health_alerts(signal_health: SignalHealthResult, alerts_cfg: AlertsConfig) -> list[Alert]:
    alerts: list[Alert] = []

    if pd.notna(signal_health.recent_ic_mean):
        if signal_health.recent_ic_mean <= alerts_cfg.signal_ic_critical:
            severity = "critical"
        elif signal_health.recent_ic_mean <= alerts_cfg.signal_ic_warning:
            severity = "warning"
        else:
            severity = None
        if severity is not None:
            alerts.append(
                Alert(
                    as_of=signal_health.as_of,
                    category=DiagnosticCategory.SIGNAL_DETERIORATION,
                    severity=severity,
                    message=(
                        f"recent IC for {signal_health.signal_name!r} ({signal_health.recent_ic_mean:.4f}) "
                        f"breached the {severity} threshold"
                    ),
                    evidence={"recent_ic_mean": signal_health.recent_ic_mean},
                )
            )

    for factor_name, drift in signal_health.exposure_drift.items():
        if drift >= alerts_cfg.exposure_drift_critical:
            severity = "critical"
        elif drift >= alerts_cfg.exposure_drift_warning:
            severity = "warning"
        else:
            continue
        alerts.append(
            Alert(
                as_of=signal_health.as_of,
                category=DiagnosticCategory.FACTOR_EXPOSURE_DRIFT,
                severity=severity,
                message=f"{factor_name!r} exposure drift ({drift:.4f}) breached the {severity} threshold",
                evidence={"drift": drift},
            )
        )

    return alerts


def check_cost_alerts(findings: Sequence[DiagnosticFinding], alerts_cfg: AlertsConfig) -> list[Alert]:
    """Promotes an existing real ``EXECUTION_COST_DETERIORATION``
    finding to a severity-tagged alert -- the finding's own real
    evidence (``relative_increase``) is reused directly, never
    recomputed.
    """
    alerts: list[Alert] = []
    for finding in findings:
        if finding.category != DiagnosticCategory.EXECUTION_COST_DETERIORATION:
            continue
        relative_increase = finding.evidence.get("relative_increase")
        if relative_increase is None:
            continue
        if relative_increase >= alerts_cfg.cost_deterioration_critical_pct:
            severity = "critical"
        elif relative_increase >= alerts_cfg.cost_deterioration_warning_pct:
            severity = "warning"
        else:
            continue
        alerts.append(
            Alert(
                as_of=finding.as_of,
                category=DiagnosticCategory.EXECUTION_COST_DETERIORATION,
                severity=severity,
                message=finding.description,
                evidence=dict(finding.evidence),
            )
        )
    return alerts


def generate_alerts(
    snapshot: StrategyPerformanceSnapshot,
    signal_health: SignalHealthResult,
    findings: Sequence[DiagnosticFinding],
    alerts_cfg: AlertsConfig,
) -> list[Alert]:
    alerts: list[Alert] = []
    alerts += check_strategy_performance_alerts(snapshot, alerts_cfg)
    alerts += check_signal_health_alerts(signal_health, alerts_cfg)
    alerts += check_cost_alerts(findings, alerts_cfg)
    return alerts
