"""End-to-end: the full Stage 9 orchestration (Stages 1-8), on a real
monthly rebalancing schedule built from factor-model-engine's own
synthetic dataset. Rebalancing is MONTHLY (matching the synthetic
dataset's own native periodicity) rather than quarterly, so this
engine's own configured windows (12, 6, 24 real periods) are
exercised against a comparably-sized real history, not a coarser one.
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
from market_regime_health.reporting import render_markdown_report, report_to_dict


def _zero_expected_returns(as_of, security_ids):
    return {sid: 0.0 for sid in security_ids}


def _standard_constraints(as_of, exposure_matrix):
    return [MaxPositionWeightConstraint(max_weight=0.2, long_only=True), GrossLeverageConstraint(max_gross_leverage=1.0)]


@pytest.fixture(scope="module")
def synthetic_fixture():
    cfg = SyntheticDataConfig(n_securities=25, n_sectors=5, n_periods_months=48)
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    dates = sorted(dataset.market["as_of"].unique())
    period_dates = [pd.Timestamp(d) for d in dates]
    rebalance_dates = period_dates[12:]

    fm_cfg = FactorModelConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    pc_cfg = PortfolioConstructionConfig()
    schedule = run_rebalancing_schedule(
        market_obs, fundamental_obs, rebalance_dates, lookback_periods=12, objective="min_variance",
        expected_returns_source=_zero_expected_returns, constraints_factory=_standard_constraints,
        fm_engine_cfg=fm_cfg, pc_engine_cfg=pc_cfg,
    )
    return market_obs, fundamental_obs, period_dates, schedule.results, fm_cfg


@pytest.fixture(scope="module")
def regime_health_result(synthetic_fixture):
    market_obs, fundamental_obs, period_dates, optimization_results, fm_cfg = synthetic_fixture
    as_of = optimization_results[-1].as_of
    engine_cfg = EngineConfig()
    return run_full_regime_health_analysis(
        market_obs, fundamental_obs, optimization_results, period_dates, as_of, "momentum",
        fm_cfg, engine_cfg, portfolio_id="demo_portfolio",
    )


def test_pipeline_produces_a_real_regime_classification_with_evidence(regime_health_result):
    r = regime_health_result
    assert r.regime_classification.as_of == r.as_of
    assert set(r.regime_classification.evidence) == {
        "realized_volatility", "trend_strength", "avg_pairwise_correlation", "liquidity_score", "n_periods_available",
    }


def test_pipeline_produces_a_real_strategy_performance_snapshot(regime_health_result):
    r = regime_health_result
    snapshot = r.strategy_performance
    assert snapshot.net_return == pytest.approx(snapshot.gross_return - snapshot.transaction_cost)
    assert -1.0 <= snapshot.drawdown <= 1e-9


def test_pipeline_produces_real_signal_health_with_decay_curve(regime_health_result):
    r = regime_health_result
    assert r.signal_health.n_periods > 0
    assert r.signal_health.decay_curve
    assert set(r.signal_health.decay_curve).issubset({1, 3, 6, 12})


def test_pipeline_history_lengths_are_internally_consistent(regime_health_result, synthetic_fixture):
    r = regime_health_result
    _, _, _, optimization_results, _ = synthetic_fixture
    assert len(r.classification_history) <= len(optimization_results)
    assert len(r.performance_history) == len(optimization_results) - 1


def test_regime_conditioned_results_cover_every_observed_regime(regime_health_result):
    r = regime_health_result
    observed_labels = {c.label for c in r.classification_history}
    reported_labels = {rc.regime for rc in r.regime_conditioned}
    assert reported_labels.issubset(observed_labels)
    for rc in r.regime_conditioned:
        assert rc.is_association_not_causation is True


def test_report_to_dict_is_json_serializable(regime_health_result):
    json.dumps(report_to_dict(regime_health_result))


def test_render_markdown_report_contains_every_real_section(regime_health_result):
    markdown = render_markdown_report(regime_health_result)
    for heading in [
        "## Market Regime", "## Strategy Performance", "## Signal Health",
        "## Diagnostic Findings", "## Alerts",
    ]:
        assert heading in markdown
    assert "synthetic" in markdown.lower()


def test_report_rendering_is_deterministic(regime_health_result):
    assert render_markdown_report(regime_health_result) == render_markdown_report(regime_health_result)
    assert json.dumps(report_to_dict(regime_health_result), sort_keys=True) == json.dumps(report_to_dict(regime_health_result), sort_keys=True)


def test_pipeline_rejects_mismatched_as_of(synthetic_fixture):
    market_obs, fundamental_obs, period_dates, optimization_results, fm_cfg = synthetic_fixture
    engine_cfg = EngineConfig()
    with pytest.raises(ValueError):
        run_full_regime_health_analysis(
            market_obs, fundamental_obs, optimization_results, period_dates, optimization_results[-2].as_of,
            "momentum", fm_cfg, engine_cfg,
        )
