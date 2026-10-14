import pandas as pd

from market_regime_health.config import RegimeConfig
from market_regime_health.data.schemas import RegimeFeatureRow, RegimeLabel
from market_regime_health.regime.classification import classify_regime

AS_OF = pd.Timestamp("2020-06-30")
CFG = RegimeConfig()


def _features(**overrides) -> RegimeFeatureRow:
    defaults = dict(
        as_of=AS_OF, realized_volatility=0.04, trend_strength=0.0, liquidity_score=0.0,
        avg_pairwise_correlation=0.10, n_securities=25, n_periods_available=CFG.min_periods_for_classification,
    )
    defaults.update(overrides)
    return RegimeFeatureRow(**defaults)


def test_insufficient_history_is_unknown():
    features = _features(n_periods_available=CFG.min_periods_for_classification - 1)
    result = classify_regime(features, CFG)
    assert result.label == RegimeLabel.UNKNOWN
    assert result.triggered_rules == ()


def test_liquidity_stress_dominates_regardless_of_other_features():
    features = _features(realized_volatility=0.01, trend_strength=0.0, liquidity_score=CFG.liquidity_stress_threshold - 0.01)
    result = classify_regime(features, CFG)
    assert result.label == RegimeLabel.LIQUIDITY_STRESSED
    assert "liquidity_stress_dominant" in result.triggered_rules


def test_high_vol_with_high_correlation_is_stressed():
    features = _features(
        realized_volatility=CFG.high_vol_threshold + 0.01, avg_pairwise_correlation=CFG.high_correlation_threshold + 0.01,
        trend_strength=0.0,
    )
    result = classify_regime(features, CFG)
    assert result.label == RegimeLabel.HIGH_VOL_STRESSED


def test_high_vol_trending_without_high_correlation():
    features = _features(
        realized_volatility=CFG.high_vol_threshold + 0.01, avg_pairwise_correlation=0.05,
        trend_strength=CFG.trend_threshold + 0.01,
    )
    result = classify_regime(features, CFG)
    assert result.label == RegimeLabel.HIGH_VOL_TRENDING


def test_low_vol_trending():
    features = _features(realized_volatility=CFG.low_vol_threshold - 0.005, trend_strength=CFG.trend_threshold + 0.01)
    result = classify_regime(features, CFG)
    assert result.label == RegimeLabel.LOW_VOL_TRENDING


def test_low_vol_range_bound():
    features = _features(realized_volatility=CFG.low_vol_threshold - 0.005, trend_strength=0.0)
    result = classify_regime(features, CFG)
    assert result.label == RegimeLabel.LOW_VOL_RANGE_BOUND


def test_moderate_vol_range_bound_is_ambiguous_not_guessed():
    midpoint = (CFG.low_vol_threshold + CFG.high_vol_threshold) / 2.0
    features = _features(realized_volatility=midpoint, trend_strength=0.0, avg_pairwise_correlation=0.05)
    result = classify_regime(features, CFG)
    assert result.label == RegimeLabel.AMBIGUOUS


def test_missing_volatility_is_ambiguous_not_guessed():
    features = _features(realized_volatility=None)
    result = classify_regime(features, CFG)
    assert result.label == RegimeLabel.AMBIGUOUS
    assert any("unavailable" in w for w in result.warnings)


def test_evidence_always_carries_every_feature_even_when_missing():
    features = _features(avg_pairwise_correlation=None)
    result = classify_regime(features, CFG)
    assert set(result.evidence) == {
        "realized_volatility", "trend_strength", "avg_pairwise_correlation", "liquidity_score", "n_periods_available",
    }
    assert pd.isna(result.evidence["avg_pairwise_correlation"])
