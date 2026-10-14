"""Stage 10 -- hardening: the comprehensive checks every engine in this
build applies. Given this engine never calls a numerical solver or a
near-zero-magnitude BLAS matrix multiply (unlike portfolio-construction-
engine's SLSQP or portfolio-risk-engine's quadratic form), exact
reproducibility is expected to hold, not just approximately.
"""

import json

import pandas as pd
import pytest
from factor_model.config import EngineConfig as FactorModelConfig, UniverseConfig
from factor_model.data import SyntheticDataConfig, fundamental_observations_from_frame, generate_synthetic_dataset, market_observations_from_frame
from portfolio_construction.config import EngineConfig as PortfolioConstructionConfig
from portfolio_construction.constraints import GrossLeverageConstraint, MaxPositionWeightConstraint
from portfolio_construction.rebalancing import run_rebalancing_schedule

from market_regime_health.config import EngineConfig
from market_regime_health.pipeline import run_full_regime_health_analysis
from market_regime_health.reporting import report_to_dict


def _assert_reports_equal(result_a, result_b) -> None:
    """Dict equality on ``==`` is wrong here: a regime observed for only
    one real period legitimately reports ``NaN`` volatility/Sharpe, and
    Python's own ``NaN != NaN`` would fail an otherwise-identical
    comparison. JSON string comparison treats two real ``NaN`` readings
    as equal text, which is what "reproducible" actually means here.
    """
    assert json.dumps(report_to_dict(result_a), sort_keys=True) == json.dumps(report_to_dict(result_b), sort_keys=True)


def _zero_expected_returns(as_of, security_ids):
    return {sid: 0.0 for sid in security_ids}


def _standard_constraints(as_of, exposure_matrix):
    return [MaxPositionWeightConstraint(max_weight=0.2, long_only=True), GrossLeverageConstraint(max_gross_leverage=1.0)]


@pytest.fixture(scope="module")
def full_synthetic_history():
    cfg = SyntheticDataConfig(n_securities=20, n_sectors=5, n_periods_months=48)
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    dates = sorted(dataset.market["as_of"].unique())
    period_dates = [pd.Timestamp(d) for d in dates]

    fm_cfg = FactorModelConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    pc_cfg = PortfolioConstructionConfig()
    schedule = run_rebalancing_schedule(
        market_obs, fundamental_obs, period_dates[12:], lookback_periods=12, objective="min_variance",
        expected_returns_source=_zero_expected_returns, constraints_factory=_standard_constraints,
        fm_engine_cfg=fm_cfg, pc_engine_cfg=pc_cfg,
    )
    return market_obs, fundamental_obs, period_dates, schedule.results, fm_cfg


def test_pipeline_is_exactly_reproducible_given_identical_inputs(full_synthetic_history):
    market_obs, fundamental_obs, period_dates, optimization_results, fm_cfg = full_synthetic_history
    as_of = optimization_results[20].as_of
    engine_cfg = EngineConfig()

    result_a = run_full_regime_health_analysis(
        market_obs, fundamental_obs, optimization_results, period_dates[: period_dates.index(as_of) + 1], as_of,
        "momentum", fm_cfg, engine_cfg,
    )
    result_b = run_full_regime_health_analysis(
        market_obs, fundamental_obs, optimization_results, period_dates[: period_dates.index(as_of) + 1], as_of,
        "momentum", fm_cfg, engine_cfg,
    )
    _assert_reports_equal(result_a, result_b)


def test_extending_the_dataset_into_the_future_never_changes_an_earlier_report(full_synthetic_history):
    """The real hardening check: querying ``as_of`` with the FULL real
    market/fundamental/optimization history available (as a live system
    actually would, with future rows already sitting in the feed) must
    give the IDENTICAL result to querying it with ONLY the data that
    existed up to ``as_of`` -- every real PIT filter inside this
    pipeline (and every function it calls) is responsible for this, not
    the caller truncating inputs by hand.
    """
    market_obs, fundamental_obs, period_dates, optimization_results, fm_cfg = full_synthetic_history
    cutoff_index = 24
    as_of = period_dates[cutoff_index]
    engine_cfg = EngineConfig()

    period_dates_truncated = period_dates[: cutoff_index + 1]
    market_obs_truncated = [obs for obs in market_obs if obs.as_of <= as_of]
    fundamental_obs_truncated = [obs for obs in fundamental_obs if obs.fundamental_period_end <= as_of]
    optimization_results_truncated = [r for r in optimization_results if r.as_of <= as_of]

    result_truncated = run_full_regime_health_analysis(
        market_obs_truncated, fundamental_obs_truncated, optimization_results_truncated, period_dates_truncated,
        as_of, "momentum", fm_cfg, engine_cfg,
    )
    result_full = run_full_regime_health_analysis(
        market_obs, fundamental_obs, optimization_results, period_dates_truncated, as_of, "momentum", fm_cfg, engine_cfg,
    )

    _assert_reports_equal(result_truncated, result_full)


def test_pipeline_handles_the_exact_minimum_history_boundary_without_crashing(full_synthetic_history):
    market_obs, fundamental_obs, period_dates, optimization_results, fm_cfg = full_synthetic_history
    engine_cfg = EngineConfig()
    min_required = engine_cfg.regime.min_periods_for_classification
    as_of = period_dates[min_required - 1 + 12]  # 12 months of lookback already consumed by the rebalancing schedule itself
    period_dates_truncated = period_dates[: period_dates.index(as_of) + 1]
    optimization_results_truncated = [r for r in optimization_results if r.as_of <= as_of]

    result = run_full_regime_health_analysis(
        market_obs, fundamental_obs, optimization_results_truncated, period_dates_truncated, as_of, "momentum",
        fm_cfg, engine_cfg,
    )
    assert result.regime_classification.as_of == as_of
