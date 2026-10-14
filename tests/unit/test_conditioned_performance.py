import pandas as pd
import pytest

from market_regime_health.data.schemas import RegimeClassification, RegimeLabel
from market_regime_health.regime_conditioned.conditioned_performance import compute_regime_conditioned_performance

DATES = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"])


def _classification(as_of, label):
    return RegimeClassification(as_of=as_of, label=label, evidence={})


def test_groups_returns_by_the_real_matching_regime_label():
    net_returns = pd.Series([0.01, 0.02, -0.01, 0.03], index=DATES)
    classifications = [
        _classification(DATES[0], RegimeLabel.LOW_VOL_TRENDING),
        _classification(DATES[1], RegimeLabel.LOW_VOL_TRENDING),
        _classification(DATES[2], RegimeLabel.HIGH_VOL_STRESSED),
        _classification(DATES[3], RegimeLabel.HIGH_VOL_STRESSED),
    ]
    results = compute_regime_conditioned_performance(net_returns, classifications, "P")
    by_label = {r.regime: r for r in results}
    assert by_label[RegimeLabel.LOW_VOL_TRENDING].n_periods == 2
    assert by_label[RegimeLabel.LOW_VOL_TRENDING].mean_net_return == pytest.approx(0.015)
    assert by_label[RegimeLabel.HIGH_VOL_STRESSED].n_periods == 2
    assert by_label[RegimeLabel.HIGH_VOL_STRESSED].mean_net_return == pytest.approx(0.01)


def test_a_date_with_no_classification_is_excluded_not_mislabeled():
    net_returns = pd.Series([0.01, 0.02], index=DATES[:2])
    classifications = [_classification(DATES[0], RegimeLabel.LOW_VOL_TRENDING)]
    results = compute_regime_conditioned_performance(net_returns, classifications, "P")
    assert len(results) == 1
    assert results[0].n_periods == 1


def test_single_period_regime_withholds_volatility_and_sharpe():
    net_returns = pd.Series([0.01], index=DATES[:1])
    classifications = [_classification(DATES[0], RegimeLabel.LOW_VOL_TRENDING)]
    results = compute_regime_conditioned_performance(net_returns, classifications, "P")
    assert results[0].sharpe is None
    assert pd.isna(results[0].volatility_of_net_return)
    assert results[0].warnings


def test_every_result_is_marked_as_association_not_causation():
    net_returns = pd.Series([0.01, 0.02], index=DATES[:2])
    classifications = [_classification(d, RegimeLabel.LOW_VOL_TRENDING) for d in DATES[:2]]
    results = compute_regime_conditioned_performance(net_returns, classifications, "P")
    assert all(r.is_association_not_causation for r in results)
