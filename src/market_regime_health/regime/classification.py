"""Stage 3 -- a real, transparent, rule-based regime classifier. Every
real feature value is always recorded in ``evidence`` (even when a rule
didn't fire on it, even when it's ``NaN`` because the feature itself was
unavailable), and every label carries the exact real rule name that
produced it -- never a black-box score. ``AMBIGUOUS`` and ``UNKNOWN``
are first-class real outcomes, not failure modes to be hidden: a
feature combination this engine's rules don't recognize is reported as
such, rather than forced into the nearest label.
"""

from __future__ import annotations

from ..config import RegimeConfig
from ..data.schemas import RegimeClassification, RegimeFeatureRow, RegimeLabel


def _vol_level(features: RegimeFeatureRow, regime_cfg: RegimeConfig) -> str | None:
    if features.realized_volatility is None:
        return None
    if features.realized_volatility > regime_cfg.high_vol_threshold:
        return "high"
    if features.realized_volatility < regime_cfg.low_vol_threshold:
        return "low"
    return "moderate"


def _trend_level(features: RegimeFeatureRow, regime_cfg: RegimeConfig) -> str | None:
    if features.trend_strength is None:
        return None
    return "trending" if abs(features.trend_strength) > regime_cfg.trend_threshold else "range_bound"


def _correlation_level(features: RegimeFeatureRow, regime_cfg: RegimeConfig) -> str | None:
    if features.avg_pairwise_correlation is None:
        return None
    return "high" if features.avg_pairwise_correlation > regime_cfg.high_correlation_threshold else "normal"


def _liquidity_level(features: RegimeFeatureRow, regime_cfg: RegimeConfig) -> str | None:
    if features.liquidity_score is None:
        return None
    return "stressed" if features.liquidity_score < regime_cfg.liquidity_stress_threshold else "normal"


def classify_regime(features: RegimeFeatureRow, regime_cfg: RegimeConfig) -> RegimeClassification:
    """Real rule priority, in order (documented here, not hidden in
    code): a real liquidity stress reading is treated as dominant
    (a liquidity crunch is dangerous regardless of how volatility or
    trend read), then high-volatility-plus-high-correlation (the real
    composite stress signal -- correlations rising together with
    volatility is the standard real "diversification breaking down"
    signature), then the remaining vol/trend combinations. Any
    combination none of those rules recognize -- including a missing
    required feature -- is ``AMBIGUOUS``, never forced into the
    nearest label.
    """
    evidence = {
        "realized_volatility": features.realized_volatility if features.realized_volatility is not None else float("nan"),
        "trend_strength": features.trend_strength if features.trend_strength is not None else float("nan"),
        "avg_pairwise_correlation": features.avg_pairwise_correlation if features.avg_pairwise_correlation is not None else float("nan"),
        "liquidity_score": features.liquidity_score if features.liquidity_score is not None else float("nan"),
        "n_periods_available": float(features.n_periods_available),
    }
    warnings = list(features.quality_flags)

    if features.n_periods_available < regime_cfg.min_periods_for_classification:
        warnings.append(
            f"only {features.n_periods_available} real periods available, below the configured "
            f"minimum of {regime_cfg.min_periods_for_classification} -- classification withheld"
        )
        return RegimeClassification(
            as_of=features.as_of, label=RegimeLabel.UNKNOWN, evidence=evidence, triggered_rules=(), warnings=tuple(warnings)
        )

    vol_level = _vol_level(features, regime_cfg)
    trend_level = _trend_level(features, regime_cfg)
    correlation_level = _correlation_level(features, regime_cfg)
    liquidity_level = _liquidity_level(features, regime_cfg)

    triggered_rules: list[str] = []
    if liquidity_level == "stressed":
        label = RegimeLabel.LIQUIDITY_STRESSED
        triggered_rules.append("liquidity_stress_dominant")
    elif vol_level == "high" and correlation_level == "high":
        label = RegimeLabel.HIGH_VOL_STRESSED
        triggered_rules.append("high_vol_high_correlation_stress")
    elif vol_level == "high" and trend_level == "trending":
        label = RegimeLabel.HIGH_VOL_TRENDING
        triggered_rules.append("high_vol_trending")
    elif vol_level == "low" and trend_level == "trending":
        label = RegimeLabel.LOW_VOL_TRENDING
        triggered_rules.append("low_vol_trending")
    elif vol_level == "low" and trend_level == "range_bound":
        label = RegimeLabel.LOW_VOL_RANGE_BOUND
        triggered_rules.append("low_vol_range_bound")
    else:
        label = RegimeLabel.AMBIGUOUS
        if vol_level is None or trend_level is None:
            warnings.append("one or more required features (volatility/trend) unavailable -- classification is ambiguous, not guessed")
        else:
            warnings.append(
                f"feature combination (vol={vol_level}, trend={trend_level}, correlation={correlation_level}) matched no real rule"
            )

    return RegimeClassification(
        as_of=features.as_of, label=label, evidence=evidence, triggered_rules=tuple(triggered_rules), warnings=tuple(warnings)
    )
