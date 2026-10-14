import pandas as pd

from market_regime_health.alerts.alerts import check_cost_alerts, check_signal_health_alerts, check_strategy_performance_alerts
from market_regime_health.config import AlertsConfig
from market_regime_health.data.schemas import DiagnosticCategory, DiagnosticFinding, SignalHealthResult, StrategyPerformanceSnapshot

AS_OF = pd.Timestamp("2020-06-30")
CFG = AlertsConfig()


def _snapshot(drawdown=0.0, rolling_sharpe=None):
    return StrategyPerformanceSnapshot(
        as_of=AS_OF, portfolio_id="P", gross_return=0.0, net_return=0.0, cumulative_net_return=0.0,
        drawdown=drawdown, rolling_sharpe=rolling_sharpe, turnover=0.1, transaction_cost=0.0,
    )


def test_drawdown_below_warning_produces_no_alert():
    alerts = check_strategy_performance_alerts(_snapshot(drawdown=-0.01), CFG)
    assert alerts == []


def test_drawdown_between_warning_and_critical_is_a_warning():
    midpoint = -(CFG.drawdown_warning + CFG.drawdown_critical) / 2.0
    alerts = check_strategy_performance_alerts(_snapshot(drawdown=midpoint), CFG)
    assert len(alerts) == 1
    assert alerts[0].severity == "warning"


def test_drawdown_beyond_critical_is_critical():
    alerts = check_strategy_performance_alerts(_snapshot(drawdown=-(CFG.drawdown_critical + 0.05)), CFG)
    assert len(alerts) == 1
    assert alerts[0].severity == "critical"


def test_rolling_sharpe_alert_uses_the_configured_thresholds():
    alerts = check_strategy_performance_alerts(_snapshot(rolling_sharpe=CFG.rolling_sharpe_critical - 0.1), CFG)
    assert any(a.severity == "critical" for a in alerts)


def test_none_rolling_sharpe_produces_no_sharpe_alert():
    alerts = check_strategy_performance_alerts(_snapshot(rolling_sharpe=None), CFG)
    assert alerts == []


def _signal_health(recent_ic_mean=0.05, exposure_drift=None):
    return SignalHealthResult(
        as_of=AS_OF, signal_name="momentum", historical_ic_mean=0.1, recent_ic_mean=recent_ic_mean, icir=0.5,
        exposure_drift=exposure_drift or {},
    )


def test_signal_ic_alert_fires_below_warning_threshold():
    alerts = check_signal_health_alerts(_signal_health(recent_ic_mean=CFG.signal_ic_warning - 0.001), CFG)
    assert any(a.category == DiagnosticCategory.SIGNAL_DETERIORATION for a in alerts)


def test_signal_ic_alert_silent_above_warning_threshold():
    alerts = check_signal_health_alerts(_signal_health(recent_ic_mean=CFG.signal_ic_warning + 0.05), CFG)
    assert alerts == []


def test_exposure_drift_alert_fires_per_factor_above_threshold():
    alerts = check_signal_health_alerts(_signal_health(exposure_drift={"momentum": 0.01, "value": CFG.exposure_drift_critical + 0.01}), CFG)
    drift_alerts = [a for a in alerts if a.category == DiagnosticCategory.FACTOR_EXPOSURE_DRIFT]
    assert len(drift_alerts) == 1
    assert drift_alerts[0].severity == "critical"


def test_cost_alert_promotes_a_matching_finding_above_critical_threshold():
    finding = DiagnosticFinding(
        as_of=AS_OF, category=DiagnosticCategory.EXECUTION_COST_DETERIORATION, description="cost up",
        evidence={"relative_increase": CFG.cost_deterioration_critical_pct + 0.1},
    )
    alerts = check_cost_alerts([finding], CFG)
    assert len(alerts) == 1
    assert alerts[0].severity == "critical"


def test_cost_alert_ignores_findings_from_other_categories():
    finding = DiagnosticFinding(as_of=AS_OF, category=DiagnosticCategory.DATA_QUALITY, description="x", evidence={"relative_increase": 1.0})
    assert check_cost_alerts([finding], CFG) == []
