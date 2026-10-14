"""Stage 6 -- real strategy performance, conditioned on (grouped by)
regime label. Reuses ``alpha_validation_toolkit.metrics.sharpe_ratio``
directly for the per-regime Sharpe -- never reimplemented. Every real
result here is an ASSOCIATION, not a causal claim (see
``data.schemas.RegimeConditionedResult.is_association_not_causation``,
always ``True``): a strategy performing poorly during a given regime
is a real correlation, never proof the regime caused it.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from quant_toolkit.metrics import sharpe_ratio

from ..data.schemas import RegimeClassification, RegimeConditionedResult, RegimeLabel


def compute_regime_conditioned_performance(
    net_returns: pd.Series,
    classifications: Sequence[RegimeClassification],
    portfolio_id: str,
) -> list[RegimeConditionedResult]:
    """Groups each real net-return observation by the real regime label
    classified for that same date, then computes real descriptive
    statistics per regime. A date with no matching classification is
    excluded -- never silently assigned a label it wasn't given.
    """
    label_by_date: dict[pd.Timestamp, RegimeLabel] = {c.as_of: c.label for c in classifications}
    returns_by_label: dict[RegimeLabel, list[float]] = {}
    for as_of, ret in net_returns.items():
        label = label_by_date.get(as_of)
        if label is None:
            continue
        returns_by_label.setdefault(label, []).append(float(ret))

    results: list[RegimeConditionedResult] = []
    for label, rets in sorted(returns_by_label.items(), key=lambda item: item[0].value):
        series = pd.Series(rets)
        n = len(series)
        warnings: list[str] = []
        mean_net_return = float(series.mean()) if n else float("nan")
        volatility_of_net_return: float
        sharpe: float | None
        if n >= 2:
            volatility_of_net_return = float(series.std(ddof=1))
            sharpe = float(sharpe_ratio(series))
        else:
            volatility_of_net_return = float("nan")
            sharpe = None
            warnings.append(f"only {n} real period(s) observed in regime {label.value!r} -- volatility/Sharpe withheld")

        results.append(
            RegimeConditionedResult(
                portfolio_id=portfolio_id,
                regime=label,
                n_periods=n,
                mean_net_return=mean_net_return,
                volatility_of_net_return=volatility_of_net_return,
                sharpe=sharpe,
                warnings=tuple(warnings),
            )
        )
    return results
