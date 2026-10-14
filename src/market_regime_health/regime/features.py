"""Stage 2 -- real market-wide regime features: volatility, trend,
liquidity, and correlation. Reuses ``factor-model-engine``'s own real
``_period_realized_returns`` (via its package-level export, the same
import path ``portfolio-risk-engine`` already uses) and
``latest_known_market`` for every real return/price lookup -- no return
computation is reimplemented here. The caller supplies a real
point-in-time universe per date (``universe_ids_by_date``, typically
built via ``factor_model.data.universe.build_point_in_time_universe``)
rather than this module recomputing universe membership itself.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from factor_model.data.point_in_time import latest_known_market
from factor_model.data.schemas import MarketObservation
from factor_model.regression import _period_realized_returns

from ..config import RegimeConfig
from ..data.schemas import RegimeFeatureRow


def _market_wide_return_series(
    market_observations: Sequence[MarketObservation],
    universe_ids_by_date: dict[pd.Timestamp, Sequence[str]],
    sorted_dates: Sequence[pd.Timestamp],
) -> pd.Series:
    """Real equal-weighted cross-sectional return at each consecutive
    pair of dates, over the real point-in-time universe known as of the
    LATER date of each pair -- the universe an investor rebalancing
    into that period would have actually used.
    """
    returns: dict[pd.Timestamp, float] = {}
    for t_prev, t_curr in zip(sorted_dates[:-1], sorted_dates[1:]):
        security_ids = universe_ids_by_date.get(t_curr, ())
        if not security_ids:
            continue
        stock_returns = _period_realized_returns(market_observations, security_ids, t_prev, t_curr)
        if len(stock_returns) == 0:
            continue
        returns[t_curr] = float(stock_returns.mean())
    return pd.Series(returns).sort_index()


def _cross_sectional_return_panel(
    market_observations: Sequence[MarketObservation],
    universe_ids_by_date: dict[pd.Timestamp, Sequence[str]],
    sorted_dates: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    """Real per-security return panel (date x security_id) for the given
    dates -- the input the correlation feature needs (co-movement
    ACROSS securities), distinct from the market-wide aggregate series
    above (which collapses the cross-section into one number per date).
    """
    panel: dict[str, dict[pd.Timestamp, float]] = {}
    for t_prev, t_curr in zip(sorted_dates[:-1], sorted_dates[1:]):
        security_ids = universe_ids_by_date.get(t_curr, ())
        stock_returns = _period_realized_returns(market_observations, security_ids, t_prev, t_curr)
        for security_id, ret in stock_returns.items():
            panel.setdefault(security_id, {})[t_curr] = ret
    return pd.DataFrame(panel).sort_index()


def _median_adv(
    market_observations: Sequence[MarketObservation], security_ids: Sequence[str], as_of: pd.Timestamp
) -> float | None:
    """Real cross-sectional median ADV among ``security_ids`` as of
    ``as_of`` -- a security with no real ADV is excluded, never
    zero-filled.
    """
    advs = [
        obs.adv_20d
        for sid in security_ids
        if (obs := latest_known_market(market_observations, sid, as_of)) is not None and obs.adv_20d is not None
    ]
    return float(np.median(advs)) if advs else None


def compute_regime_features(
    market_observations: Sequence[MarketObservation],
    universe_ids_by_date: dict[pd.Timestamp, Sequence[str]],
    period_dates: Sequence[pd.Timestamp],
    as_of: pd.Timestamp,
    regime_cfg: RegimeConfig,
) -> RegimeFeatureRow:
    """``period_dates`` must have ``as_of`` as its last (sorted) real
    element -- the same real PIT discipline ``portfolio-risk-engine``'s
    own pipeline enforces: never uses a date after ``as_of``.
    """
    sorted_dates = sorted(period_dates)
    if not sorted_dates or sorted_dates[-1] != as_of:
        raise ValueError("as_of must be the last real date in period_dates (no look-ahead)")

    market_return_series = _market_wide_return_series(market_observations, universe_ids_by_date, sorted_dates)
    n_periods_available = len(market_return_series)
    n_securities = len(universe_ids_by_date.get(as_of, ()))
    quality_flags: list[str] = []

    realized_volatility: float | None = None
    trend_strength: float | None = None
    if n_periods_available >= 2:
        vol_window = market_return_series.tail(regime_cfg.volatility_window)
        if len(vol_window) >= 2:
            realized_volatility = float(vol_window.std(ddof=1))
        trend_window_series = market_return_series.tail(regime_cfg.trend_window)
        if len(trend_window_series) >= 1:
            trend_strength = float((1.0 + trend_window_series).prod() - 1.0)
    else:
        quality_flags.append("insufficient_return_history")

    avg_pairwise_correlation: float | None = None
    correlation_dates = sorted_dates[-(regime_cfg.correlation_window + 1):]
    return_panel = _cross_sectional_return_panel(market_observations, universe_ids_by_date, correlation_dates)
    if return_panel.shape[1] >= 2 and return_panel.shape[0] >= 2:
        corr_matrix = return_panel.corr().to_numpy()
        n = corr_matrix.shape[0]
        off_diagonal_sum = corr_matrix.sum() - np.trace(corr_matrix)
        avg_pairwise_correlation = float(off_diagonal_sum / (n * (n - 1)))
    else:
        quality_flags.append("insufficient_cross_section_for_correlation")

    liquidity_score: float | None = None
    liquidity_dates = sorted_dates[-regime_cfg.liquidity_window :]
    if len(liquidity_dates) >= 2:
        start_adv = _median_adv(market_observations, universe_ids_by_date.get(liquidity_dates[0], ()), liquidity_dates[0])
        end_adv = _median_adv(market_observations, universe_ids_by_date.get(liquidity_dates[-1], ()), liquidity_dates[-1])
        if start_adv is not None and end_adv is not None and start_adv > 0:
            liquidity_score = float(end_adv / start_adv - 1.0)
        else:
            quality_flags.append("missing_adv_for_liquidity_feature")
    else:
        quality_flags.append("insufficient_history_for_liquidity_feature")

    return RegimeFeatureRow(
        as_of=as_of,
        realized_volatility=realized_volatility,
        trend_strength=trend_strength,
        liquidity_score=liquidity_score,
        avg_pairwise_correlation=avg_pairwise_correlation,
        n_securities=n_securities,
        n_periods_available=n_periods_available,
        quality_flags=tuple(quality_flags),
    )
