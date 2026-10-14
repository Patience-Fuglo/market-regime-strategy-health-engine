import pandas as pd
import pytest
from factor_model.data.schemas import MarketObservation, Sector

from market_regime_health.config import RegimeConfig
from market_regime_health.regime.features import compute_regime_features

DATES = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"])
# A: +10%, -10%, +10% ; B: -10%, +10%, -10% (perfectly anti-correlated)
PRICES_A = [100.0, 110.0, 99.0, 108.9]
PRICES_B = [100.0, 90.0, 99.0, 89.1]
ADV_A = [1000.0, 1000.0, 1000.0, 800.0]
ADV_B = [1000.0, 1000.0, 1000.0, 800.0]


def _obs(security_id, as_of, price, adv):
    return MarketObservation(
        security_id=security_id, ticker=security_id, as_of=as_of, known_as_of=as_of, sector=Sector.TECHNOLOGY,
        is_active=True, price=price, shares_outstanding=1.0, market_cap=price, adv_20d=adv,
    )


def _market_observations():
    obs = []
    for i, d in enumerate(DATES):
        obs.append(_obs("A", d, PRICES_A[i], ADV_A[i]))
        obs.append(_obs("B", d, PRICES_B[i], ADV_B[i]))
    return obs


def _cfg(**overrides):
    defaults = dict(volatility_window=3, trend_window=3, correlation_window=3, liquidity_window=2)
    defaults.update(overrides)
    return RegimeConfig(**defaults)


def test_trend_and_volatility_from_a_perfectly_offsetting_return_series():
    market_obs = _market_observations()
    universe_ids_by_date = {d: ["A", "B"] for d in DATES}
    features = compute_regime_features(market_obs, universe_ids_by_date, DATES, DATES[-1], _cfg())
    assert features.n_periods_available == 3
    assert features.trend_strength == pytest.approx(0.0, abs=1e-9)
    assert features.realized_volatility == pytest.approx(0.0, abs=1e-9)


def test_avg_pairwise_correlation_of_two_anti_correlated_securities():
    market_obs = _market_observations()
    universe_ids_by_date = {d: ["A", "B"] for d in DATES}
    features = compute_regime_features(market_obs, universe_ids_by_date, DATES, DATES[-1], _cfg())
    assert features.avg_pairwise_correlation == pytest.approx(-1.0, abs=1e-6)


def test_liquidity_score_reflects_the_real_relative_adv_change():
    market_obs = _market_observations()
    universe_ids_by_date = {d: ["A", "B"] for d in DATES}
    features = compute_regime_features(market_obs, universe_ids_by_date, DATES, DATES[-1], _cfg())
    assert features.liquidity_score == pytest.approx(800.0 / 1000.0 - 1.0)


def test_rejects_an_as_of_that_is_not_the_last_real_date():
    market_obs = _market_observations()
    universe_ids_by_date = {d: ["A", "B"] for d in DATES}
    with pytest.raises(ValueError):
        compute_regime_features(market_obs, universe_ids_by_date, DATES, DATES[-2], _cfg())


def test_single_security_universe_withholds_correlation():
    market_obs = _market_observations()
    universe_ids_by_date = {d: ["A"] for d in DATES}
    features = compute_regime_features(market_obs, universe_ids_by_date, DATES, DATES[-1], _cfg())
    assert features.avg_pairwise_correlation is None
    assert "insufficient_cross_section_for_correlation" in features.quality_flags
