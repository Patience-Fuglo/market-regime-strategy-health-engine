import pandas as pd
from factor_model.data.schemas import ExposureSubject, FactorExposureRow

from market_regime_health.strategy_health.signal_health import _exposure_drift

AS_OF = pd.Timestamp("2020-06-30")
PRIOR = pd.Timestamp("2020-05-31")


def _row(as_of, factor_name, value, portfolio_id="P"):
    return FactorExposureRow(
        subject_type=ExposureSubject.PORTFOLIO, subject_id=portfolio_id, as_of=as_of, factor_name=factor_name,
        exposure_value=value, source_known_as_of=as_of,
    )


def test_drift_is_the_real_absolute_change_between_the_two_most_recent_snapshots():
    rows = [_row(PRIOR, "momentum", 0.20), _row(AS_OF, "momentum", 0.50)]
    drift = _exposure_drift(rows, "P", AS_OF)
    assert drift == {"momentum": 0.30}


def test_a_factor_missing_in_one_snapshot_is_treated_as_a_real_zero():
    rows = [_row(PRIOR, "momentum", 0.20), _row(AS_OF, "momentum", 0.20), _row(AS_OF, "value", 0.10)]
    drift = _exposure_drift(rows, "P", AS_OF)
    assert drift == {"momentum": 0.0, "value": 0.10}


def test_fewer_than_two_snapshots_returns_empty_not_fabricated_zero():
    rows = [_row(AS_OF, "momentum", 0.20)]
    assert _exposure_drift(rows, "P", AS_OF) == {}


def test_other_portfolios_and_subject_types_are_ignored():
    rows = [
        _row(PRIOR, "momentum", 0.20, portfolio_id="OTHER"),
        _row(AS_OF, "momentum", 0.50, portfolio_id="OTHER"),
        FactorExposureRow(subject_type=ExposureSubject.STOCK, subject_id="AAA", as_of=PRIOR, factor_name="momentum", exposure_value=1.0, source_known_as_of=PRIOR),
    ]
    assert _exposure_drift(rows, "P", AS_OF) == {}
