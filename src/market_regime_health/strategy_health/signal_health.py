"""Stage 5 -- real signal health monitoring. Reuses
``factor_model.monitoring.compute_factor_stability`` directly for the
primary real IC/ICIR/drift diagnostic -- the same real, point-in-time,
cross-sectional lagged-exposure-vs-realized-return computation that
module already performs (reusing Stage 5/6's own exact per-period fit),
including its own real drift/sign-flip warnings, carried through
unchanged rather than re-derived with a second, parallel threshold.

This module adds two real capabilities ``factor_model.monitoring``
does not provide:

- a real multi-horizon decay curve: the SAME real cross-sectional IC
  idea (lagged exposure vs. realized return), generalized across
  several forward horizons instead of just one period ahead, using
  ``factor_model``'s own real ``compute_factor_exposure_table``/
  ``_period_realized_returns`` and ``alpha_validation_toolkit``'s real
  ``information_coefficient`` -- none of that math is reimplemented,
  only recombined at a different horizon.
- real portfolio-level exposure drift: the change in a specific
  portfolio's own real factor exposure vector
  (``factor_model.data.schemas.FactorExposureRow``, subject_type
  PORTFOLIO) between two consecutive real snapshots -- a genuinely
  different real question ("is my portfolio's own conviction
  shifting") from factor-wide predictive-power stability.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from factor_model.config import EngineConfig as FactorModelConfig
from factor_model.data.schemas import ExposureSubject, FactorExposureRow, FundamentalObservation, MarketObservation, UniversePITSnapshot
from factor_model.factors import compute_factor_exposure_table
from factor_model.monitoring import compute_factor_stability
from factor_model.regression import _period_realized_returns
from quant_toolkit.metrics import information_coefficient

from ..config import SignalHealthConfig
from ..data.schemas import SignalHealthResult


def _exposure_drift(exposure_rows: Sequence[FactorExposureRow], portfolio_id: str, as_of: pd.Timestamp) -> dict[str, float]:
    """Real |change| in each real factor's exposure between the two
    most recent real portfolio-level snapshots at or before ``as_of``.
    Empty when fewer than two real snapshots exist yet -- never a
    fabricated zero-drift reading.
    """
    own_rows = [
        row
        for row in exposure_rows
        if row.subject_type == ExposureSubject.PORTFOLIO and row.subject_id == portfolio_id and row.as_of <= as_of
    ]
    dates = sorted({row.as_of for row in own_rows})
    if len(dates) < 2:
        return {}
    prev_date, curr_date = dates[-2], dates[-1]
    prev_by_factor = {row.factor_name: row.exposure_value for row in own_rows if row.as_of == prev_date}
    curr_by_factor = {row.factor_name: row.exposure_value for row in own_rows if row.as_of == curr_date}
    factor_names = set(prev_by_factor) | set(curr_by_factor)
    return {
        factor_name: abs(curr_by_factor.get(factor_name, 0.0) - prev_by_factor.get(factor_name, 0.0))
        for factor_name in sorted(factor_names)
    }


def _decay_curve(
    market_observations: Sequence[MarketObservation],
    fundamental_observations: Sequence[FundamentalObservation],
    universe_snapshot_by_date: dict[pd.Timestamp, UniversePITSnapshot],
    sorted_dates: Sequence[pd.Timestamp],
    factor_name: str,
    fm_engine_cfg: FactorModelConfig,
    horizons: Sequence[int],
) -> dict[int, float]:
    """For each real horizon ``h``, the mean real cross-sectional IC
    between a security's real lagged exposure at ``t_prev`` and its
    real realized return from ``t_prev`` to the date ``h`` real periods
    later -- at ``h=1`` this is the same real quantity
    ``compute_factor_stability``'s own ``historical_ic_mean`` measures;
    larger ``h`` answers "how many periods can this exposure reading be
    acted on before it goes stale."
    """
    decay: dict[int, float] = {}
    for h in horizons:
        ic_values: list[float] = []
        for i in range(len(sorted_dates) - h):
            t_prev, t_target = sorted_dates[i], sorted_dates[i + h]
            stock_exposure_rows, _ = compute_factor_exposure_table(
                market_observations, fundamental_observations, t_prev, universe_snapshot_by_date[t_prev], fm_engine_cfg
            )
            exposure_series = pd.Series(
                {row.subject_id: row.exposure_value for row in stock_exposure_rows if row.factor_name == factor_name}
            )
            if len(exposure_series) < 2:
                continue
            realized_returns = _period_realized_returns(market_observations, list(exposure_series.index), t_prev, t_target)
            if len(realized_returns) < 2:
                continue
            ic_value = information_coefficient(exposure_series, realized_returns)
            if pd.notna(ic_value):
                ic_values.append(ic_value)
        if ic_values:
            decay[h] = float(sum(ic_values) / len(ic_values))
    return decay


def compute_signal_health(
    market_observations: Sequence[MarketObservation],
    fundamental_observations: Sequence[FundamentalObservation],
    universe_snapshot_by_date: dict[pd.Timestamp, UniversePITSnapshot],
    exposure_rows: Sequence[FactorExposureRow],
    sorted_dates: Sequence[pd.Timestamp],
    portfolio_id: str,
    factor_name: str,
    as_of: pd.Timestamp,
    fm_engine_cfg: FactorModelConfig,
    signal_health_cfg: SignalHealthConfig,
) -> SignalHealthResult:
    """``sorted_dates`` must already be filtered to real dates
    ``<= as_of`` by the caller -- this function (like
    ``compute_factor_stability`` itself) trusts whatever date range
    it's given, so the PIT discipline lives one level up, in the
    pipeline that assembles this real date range.
    """
    stability_by_factor = compute_factor_stability(market_observations, fundamental_observations, sorted_dates, fm_engine_cfg)
    warnings: list[str] = []
    if factor_name not in stability_by_factor:
        warnings.append(f"no real factor-stability result for {factor_name!r} -- insufficient history or an unrecognized factor name")
        return SignalHealthResult(
            as_of=as_of, signal_name=factor_name, historical_ic_mean=float("nan"), recent_ic_mean=float("nan"),
            icir=float("nan"), decay_curve={}, exposure_drift=_exposure_drift(exposure_rows, portfolio_id, as_of),
            n_periods=0, warnings=tuple(warnings),
        )

    stability = stability_by_factor[factor_name]
    warnings.extend(stability.warnings)
    if stability.ic_series.notna().sum() < signal_health_cfg.min_observations_for_reliable_ic:
        warnings.append(
            f"only {int(stability.ic_series.notna().sum())} real periods of IC history, below the configured "
            f"minimum of {signal_health_cfg.min_observations_for_reliable_ic} -- IC is unreliable, not withheld"
        )

    decay = _decay_curve(
        market_observations, fundamental_observations, universe_snapshot_by_date, sorted_dates, factor_name,
        fm_engine_cfg, signal_health_cfg.decay_horizons,
    )
    if not decay:
        warnings.append("insufficient real history to compute a decay curve at any configured horizon")

    return SignalHealthResult(
        as_of=as_of,
        signal_name=factor_name,
        historical_ic_mean=stability.historical_ic_mean,
        recent_ic_mean=stability.recent_ic_mean,
        icir=stability.icir,
        decay_curve=decay,
        exposure_drift=_exposure_drift(exposure_rows, portfolio_id, as_of),
        n_periods=int(stability.ic_series.notna().sum()),
        warnings=tuple(warnings),
    )
