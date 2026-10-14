import pytest

from market_regime_health.config import AlertsConfig, EngineConfig, RegimeConfig, SignalHealthConfig, StrategyHealthConfig, load_config


def test_default_engine_config_constructs_cleanly():
    cfg = EngineConfig()
    assert cfg.regime.volatility_window == 12


def test_regime_config_rejects_low_vol_above_high_vol():
    with pytest.raises(ValueError):
        RegimeConfig(low_vol_threshold=0.10, high_vol_threshold=0.05)


def test_regime_config_rejects_tiny_windows():
    with pytest.raises(ValueError):
        RegimeConfig(volatility_window=1)


def test_strategy_health_config_rejects_tiny_window():
    with pytest.raises(ValueError):
        StrategyHealthConfig(rolling_sharpe_window=1)


def test_signal_health_config_rejects_empty_decay_horizons():
    with pytest.raises(ValueError):
        SignalHealthConfig(decay_horizons=())


def test_signal_health_config_rejects_non_positive_horizon():
    with pytest.raises(ValueError):
        SignalHealthConfig(decay_horizons=(1, 0, 3))


def test_alerts_config_rejects_inverted_drawdown_thresholds():
    with pytest.raises(ValueError):
        AlertsConfig(drawdown_warning=0.30, drawdown_critical=0.10)


def test_load_config_from_the_real_default_yaml():
    cfg = load_config("configs/default.yaml")
    assert cfg.regime.high_vol_threshold == pytest.approx(0.06)
    assert cfg.signal_health.decay_horizons == (1, 3, 6, 12)
