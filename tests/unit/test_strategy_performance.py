import pandas as pd
import pytest
from portfolio_construction.backtest import BacktestResult
from portfolio_construction.data import OptimizationResult

from market_regime_health.config import StrategyHealthConfig
from market_regime_health.strategy_health.performance import build_strategy_performance_snapshots

DATES = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"])


def _opt_result(as_of, turnover):
    return OptimizationResult(
        as_of=as_of, portfolio_id="P", target_weights={"A": 1.0}, expected_return=0.0, expected_variance=0.0,
        turnover=turnover, converged=True, transaction_cost=0.0,
    )


def _backtest_result(net_return_values):
    net_returns = pd.Series(net_return_values, index=DATES[1:])
    return BacktestResult(
        gross_returns=net_returns.copy(), transaction_costs=pd.Series(0.0, index=net_returns.index),
        net_returns=net_returns, cumulative_net_return=float((1.0 + net_returns).prod() - 1.0),
    )


def test_drawdown_recovers_a_known_synthetic_path():
    # +10%, -20%, +10%: wealth path 1.10, 0.88, 0.968 -- peak is 1.10, so
    # the real drawdown at the third period is 0.968/1.10 - 1 = -0.12
    backtest = _backtest_result([0.10, -0.20, 0.10])
    optimization_results = [_opt_result(d, turnover=0.1) for d in DATES]
    snapshots = build_strategy_performance_snapshots(backtest, optimization_results, "P", StrategyHealthConfig())
    assert snapshots[0].drawdown == pytest.approx(0.0)
    assert snapshots[1].drawdown == pytest.approx(-0.20)
    assert snapshots[2].drawdown == pytest.approx(0.968 / 1.10 - 1.0)


def test_cumulative_net_return_matches_the_real_wealth_index():
    backtest = _backtest_result([0.10, -0.20, 0.10])
    optimization_results = [_opt_result(d, turnover=0.1) for d in DATES]
    snapshots = build_strategy_performance_snapshots(backtest, optimization_results, "P", StrategyHealthConfig())
    assert snapshots[-1].cumulative_net_return == pytest.approx(0.968 - 1.0)


def test_rolling_sharpe_withheld_below_min_periods():
    backtest = _backtest_result([0.01, 0.02, -0.01])
    optimization_results = [_opt_result(d, turnover=0.1) for d in DATES]
    cfg = StrategyHealthConfig(rolling_sharpe_window=12, min_periods_for_rolling_sharpe=6)
    snapshots = build_strategy_performance_snapshots(backtest, optimization_results, "P", cfg)
    assert all(s.rolling_sharpe is None for s in snapshots)
    assert any("rolling Sharpe withheld" in w for s in snapshots for w in s.warnings)


def test_turnover_is_attributed_to_the_establishing_rebalance():
    backtest = _backtest_result([0.01, 0.02, -0.01])
    optimization_results = [_opt_result(DATES[0], turnover=0.5), _opt_result(DATES[1], turnover=0.3), _opt_result(DATES[2], turnover=0.2), _opt_result(DATES[3], turnover=0.1)]
    snapshots = build_strategy_performance_snapshots(backtest, optimization_results, "P", StrategyHealthConfig())
    # the holding period ending at DATES[1] was established by the rebalance at DATES[0]
    assert snapshots[0].turnover == pytest.approx(0.5)
    assert snapshots[1].turnover == pytest.approx(0.3)
    assert snapshots[2].turnover == pytest.approx(0.2)
