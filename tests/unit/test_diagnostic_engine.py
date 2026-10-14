import pandas as pd
from factor_model.diagnostics.data_health import DataHealthReport

from market_regime_health.config import DiagnosticsConfig
from market_regime_health.data.schemas import DiagnosticCategory, RegimeClassification, RegimeLabel, SignalHealthResult, StrategyPerformanceSnapshot
from market_regime_health.diagnostics.diagnostic_engine import (
    diagnose_cost_deterioration,
    diagnose_data_quality,
    diagnose_exposure_drift,
    diagnose_regime_change,
    diagnose_signal_deterioration,
)

AS_OF = pd.Timestamp("2020-06-30")
PRIOR = pd.Timestamp("2020-05-31")


def test_data_quality_findings_mirror_every_real_warning():
    report = DataHealthReport(as_of=AS_OF, n_securities_observed=25, n_duplicate_market_rows=0, n_duplicate_fundamental_vintages=0, warnings=["w1", "w2"])
    findings = diagnose_data_quality(AS_OF, report)
    assert [f.description for f in findings] == ["w1", "w2"]
    assert all(f.category == DiagnosticCategory.DATA_QUALITY and not f.is_causal_claim for f in findings)


def test_signal_deterioration_mirrors_factor_model_s_own_warnings():
    signal_health = SignalHealthResult(as_of=AS_OF, signal_name="momentum", historical_ic_mean=0.1, recent_ic_mean=0.01, icir=0.5, warnings=("decayed",))
    findings = diagnose_signal_deterioration(signal_health)
    assert len(findings) == 1
    assert findings[0].category == DiagnosticCategory.SIGNAL_DETERIORATION
    assert findings[0].description == "decayed"


def test_exposure_drift_only_flags_factors_above_threshold():
    cfg = DiagnosticsConfig(exposure_drift_threshold=0.15)
    signal_health = SignalHealthResult(
        as_of=AS_OF, signal_name="momentum", historical_ic_mean=0.1, recent_ic_mean=0.1, icir=0.5,
        exposure_drift={"momentum": 0.05, "value": 0.30},
    )
    findings = diagnose_exposure_drift(signal_health, cfg)
    assert len(findings) == 1
    assert "value" in findings[0].description


def test_regime_change_fires_only_when_the_label_actually_changed():
    prior = RegimeClassification(as_of=PRIOR, label=RegimeLabel.LOW_VOL_TRENDING, evidence={})
    same = RegimeClassification(as_of=AS_OF, label=RegimeLabel.LOW_VOL_TRENDING, evidence={})
    changed = RegimeClassification(as_of=AS_OF, label=RegimeLabel.HIGH_VOL_STRESSED, evidence={})

    assert diagnose_regime_change(same, prior) == []
    findings = diagnose_regime_change(changed, prior)
    assert len(findings) == 1
    assert findings[0].category == DiagnosticCategory.MARKET_REGIME_CHANGE
    assert not findings[0].is_causal_claim


def test_cost_deterioration_fires_only_above_the_configured_relative_increase():
    cfg = DiagnosticsConfig(cost_deterioration_pct_threshold=0.25)
    snapshots = [
        StrategyPerformanceSnapshot(as_of=PRIOR, portfolio_id="P", gross_return=0.0, net_return=0.0, cumulative_net_return=0.0, drawdown=0.0, rolling_sharpe=None, turnover=0.1, transaction_cost=0.01),
        StrategyPerformanceSnapshot(as_of=AS_OF, portfolio_id="P", gross_return=0.0, net_return=0.0, cumulative_net_return=0.0, drawdown=0.0, rolling_sharpe=None, turnover=0.1, transaction_cost=0.02),
    ]
    findings = diagnose_cost_deterioration(snapshots, cfg)
    assert len(findings) == 1
    assert findings[0].evidence["relative_increase"] == 1.0


def test_cost_deterioration_silent_when_below_threshold():
    cfg = DiagnosticsConfig(cost_deterioration_pct_threshold=0.25)
    snapshots = [
        StrategyPerformanceSnapshot(as_of=PRIOR, portfolio_id="P", gross_return=0.0, net_return=0.0, cumulative_net_return=0.0, drawdown=0.0, rolling_sharpe=None, turnover=0.1, transaction_cost=0.01),
        StrategyPerformanceSnapshot(as_of=AS_OF, portfolio_id="P", gross_return=0.0, net_return=0.0, cumulative_net_return=0.0, drawdown=0.0, rolling_sharpe=None, turnover=0.1, transaction_cost=0.011),
    ]
    assert diagnose_cost_deterioration(snapshots, cfg) == []
