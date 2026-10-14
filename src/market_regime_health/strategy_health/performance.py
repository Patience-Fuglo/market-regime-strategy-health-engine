"""Stage 4 -- real strategy performance monitoring. Gross/net returns,
turnover, and transaction cost are READ directly from
``portfolio-construction-engine``'s own real ``BacktestResult``/
``OptimizationResult`` -- never recomputed. This module's own real
additions are drawdown (derived from the real net-return series via
the standard wealth-index formula) and rolling Sharpe
(``alpha_validation_toolkit.metrics.sharpe_ratio`` applied to each real
trailing window -- the Sharpe formula itself is never reimplemented).
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from portfolio_construction.backtest import BacktestResult
from portfolio_construction.data import OptimizationResult
from quant_toolkit.metrics import sharpe_ratio

from ..config import StrategyHealthConfig
from ..data.schemas import StrategyPerformanceSnapshot


def _turnover_by_period_end(optimization_results: Sequence[OptimizationResult]) -> dict[pd.Timestamp, float]:
    """Mirrors ``run_backtest``'s own real pairing convention: the
    turnover incurred ESTABLISHING a rebalance's weights is attributed
    to the holding period those weights cover, i.e. to the NEXT
    rebalance's ``as_of`` -- the same real attribution ``run_backtest``
    already uses for transaction cost. ``BacktestResult`` does not
    itself expose per-period turnover, only transaction cost, so this
    mirrors that exact pairing rather than inventing a new one.
    """
    sorted_results = sorted(optimization_results, key=lambda r: r.as_of)
    return {end.as_of: start.turnover for start, end in zip(sorted_results[:-1], sorted_results[1:])}


def build_strategy_performance_snapshots(
    backtest_result: BacktestResult,
    optimization_results: Sequence[OptimizationResult],
    portfolio_id: str,
    strategy_health_cfg: StrategyHealthConfig,
) -> list[StrategyPerformanceSnapshot]:
    """One real ``StrategyPerformanceSnapshot`` per date in
    ``backtest_result.net_returns`` -- the real holding-period dates
    the backtest already measured, never a new date grid.
    """
    net_returns = backtest_result.net_returns
    gross_returns = backtest_result.gross_returns
    transaction_costs = backtest_result.transaction_costs
    turnover_by_period_end = _turnover_by_period_end(optimization_results)

    wealth_index = (1.0 + net_returns).cumprod()
    running_peak = wealth_index.cummax()
    drawdown_series = wealth_index / running_peak - 1.0
    cumulative_series = wealth_index - 1.0

    snapshots: list[StrategyPerformanceSnapshot] = []
    for i, as_of in enumerate(net_returns.index):
        warnings: list[str] = []
        rolling_sharpe: float | None = None
        trailing = net_returns.iloc[max(0, i + 1 - strategy_health_cfg.rolling_sharpe_window) : i + 1]
        if len(trailing) >= strategy_health_cfg.min_periods_for_rolling_sharpe:
            rolling_sharpe = float(sharpe_ratio(trailing))
        else:
            warnings.append(
                f"only {len(trailing)} real trailing periods available, below the configured minimum of "
                f"{strategy_health_cfg.min_periods_for_rolling_sharpe} -- rolling Sharpe withheld"
            )

        turnover = turnover_by_period_end.get(as_of)
        if turnover is None:
            warnings.append(f"no real matching OptimizationResult pairing found for {as_of.date()} -- turnover unavailable")
            turnover = float("nan")

        snapshots.append(
            StrategyPerformanceSnapshot(
                as_of=as_of,
                portfolio_id=portfolio_id,
                gross_return=float(gross_returns.loc[as_of]),
                net_return=float(net_returns.loc[as_of]),
                cumulative_net_return=float(cumulative_series.loc[as_of]),
                drawdown=float(drawdown_series.loc[as_of]),
                rolling_sharpe=rolling_sharpe,
                turnover=turnover,
                transaction_cost=float(transaction_costs.loc[as_of]),
                warnings=tuple(warnings),
            )
        )
    return snapshots
