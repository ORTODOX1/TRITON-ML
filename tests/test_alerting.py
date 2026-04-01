"""Tests for AlertEngine severity evaluation logic."""

from __future__ import annotations

import pytest

from triton_ml.alerting import AlertEngine, Severity, Alert
from triton_ml.config import Settings, AlertThresholds


@pytest.fixture
def engine() -> AlertEngine:
    """AlertEngine with default thresholds."""
    return AlertEngine(settings=Settings())


@pytest.fixture
def custom_engine() -> AlertEngine:
    """AlertEngine with explicit thresholds for precise testing."""
    settings = Settings(
        alerts=AlertThresholds(
            watch_rul_hours=720.0,
            alarm_rul_hours=168.0,
            shutdown_rul_hours=24.0,
            anomaly_score_limit=-0.35,
        )
    )
    return AlertEngine(settings=settings)


class TestSeverityEscalation:
    """Verify correct tier assignment based on RUL hours."""

    def test_normal_when_rul_high(self, engine: AlertEngine) -> None:
        alert = engine.evaluate("ME-001", rul_hours=1000.0, anomaly_score=0.1, fault_label="NORMAL")
        assert alert.severity == Severity.NORMAL

    def test_watch_within_threshold(self, custom_engine: AlertEngine) -> None:
        alert = custom_engine.evaluate("ME-001", rul_hours=500.0, anomaly_score=0.1, fault_label="BEARING_WEAR")
        assert alert.severity == Severity.WATCH

    def test_alarm_within_threshold(self, custom_engine: AlertEngine) -> None:
        alert = custom_engine.evaluate("ME-001", rul_hours=100.0, anomaly_score=0.1, fault_label="MISALIGNMENT")
        assert alert.severity == Severity.ALARM

    def test_shutdown_below_threshold(self, custom_engine: AlertEngine) -> None:
        alert = custom_engine.evaluate("ME-001", rul_hours=10.0, anomaly_score=0.1, fault_label="BEARING_WEAR")
        assert alert.severity == Severity.SHUTDOWN


class TestSeverityTransitions:
    """NORMAL -> WATCH -> ALARM -> SHUTDOWN as RUL decreases."""

    def test_progressive_degradation(self, custom_engine: AlertEngine) -> None:
        rul_values = [2000.0, 500.0, 100.0, 10.0]
        expected = [Severity.NORMAL, Severity.WATCH, Severity.ALARM, Severity.SHUTDOWN]

        for rul, expected_sev in zip(rul_values, expected):
            alert = custom_engine.evaluate("PUMP-003", rul_hours=rul, anomaly_score=0.1, fault_label="NORMAL")
            assert alert.severity == expected_sev, f"RUL={rul} expected {expected_sev.name}"

    def test_anomaly_score_escalates_to_alarm(self, custom_engine: AlertEngine) -> None:
        """A low anomaly score should override NORMAL/WATCH up to ALARM."""
        alert = custom_engine.evaluate(
            "TC-002", rul_hours=2000.0, anomaly_score=-0.5, fault_label="FOULING"
        )
        assert alert.severity == Severity.ALARM


class TestAlertPayload:
    """Verify the Alert dataclass fields are populated correctly."""

    def test_alert_contains_equipment_id(self, engine: AlertEngine) -> None:
        alert = engine.evaluate("GEN-01", rul_hours=50.0, anomaly_score=0.0, fault_label="IMBALANCE")
        assert alert.equipment_id == "GEN-01"

    def test_alert_message_includes_severity(self, engine: AlertEngine) -> None:
        alert = engine.evaluate("ME-001", rul_hours=10.0, anomaly_score=0.0, fault_label="BEARING_WEAR")
        assert "SHUTDOWN" in alert.message

    def test_alert_timestamp_is_iso_format(self, engine: AlertEngine) -> None:
        alert = engine.evaluate("ME-001", rul_hours=500.0, anomaly_score=0.0, fault_label="NORMAL")
        # ISO format contains 'T' separator and timezone info
        assert "T" in alert.timestamp


class TestAlertDeduplication:
    """Identical conditions should produce identical severity -- no drift."""

    def test_repeated_evaluation_same_result(self, engine: AlertEngine) -> None:
        alerts = [
            engine.evaluate("ME-001", rul_hours=100.0, anomaly_score=0.0, fault_label="FOULING")
            for _ in range(5)
        ]
        severities = {a.severity for a in alerts}
        assert len(severities) == 1, "Repeated evaluation must produce consistent severity"
