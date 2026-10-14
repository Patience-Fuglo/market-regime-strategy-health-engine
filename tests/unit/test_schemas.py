import pandas as pd
import pytest

from market_regime_health.data.schemas import Alert, DiagnosticCategory


def test_alert_rejects_an_unknown_severity():
    with pytest.raises(ValueError):
        Alert(as_of=pd.Timestamp("2020-01-31"), category=DiagnosticCategory.DATA_QUALITY, severity="urgent", message="x", evidence={})


def test_alert_accepts_warning_and_critical():
    for severity in ("warning", "critical"):
        Alert(as_of=pd.Timestamp("2020-01-31"), category=DiagnosticCategory.DATA_QUALITY, severity=severity, message="x", evidence={})
