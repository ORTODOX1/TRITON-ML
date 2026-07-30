"""
Four-tier alert engine for machinery health status.

Pushes alerts over a WebSocket connection to a monitoring dashboard.
The four severity tiers are a project convention, not a class society
or IMO requirement.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import IntEnum

import websockets.sync.client as ws_client

from triton_ml.config import Settings

logger = logging.getLogger(__name__)


class Severity(IntEnum):
    """Alert tiers aligned with class society condition monitoring rules."""
    NORMAL = 0
    WATCH = 1
    ALARM = 2
    SHUTDOWN = 3


@dataclass
class Alert:
    """Structured alert payload for AEGIS-MONITOR ingestion."""
    equipment_id: str
    severity: Severity
    rul_hours: float
    anomaly_score: float
    fault_label: str
    timestamp: str
    message: str


class AlertEngine:
    """Evaluate model outputs and emit alerts over WebSocket."""

    def __init__(self, settings: Settings | None = None,
                 ws_url: str = "ws://aegis-monitor:8400/ws/alerts") -> None:
        self._cfg = settings or Settings()
        self._ws_url = ws_url

    def evaluate(self, equipment_id: str, rul_hours: float,
                 anomaly_score: float, fault_label: str) -> Alert:
        """Determine severity tier from RUL and anomaly score."""
        thresholds = self._cfg.alerts
        if rul_hours <= thresholds.shutdown_rul_hours:
            severity = Severity.SHUTDOWN
        elif rul_hours <= thresholds.alarm_rul_hours:
            severity = Severity.ALARM
        elif rul_hours <= thresholds.watch_rul_hours:
            severity = Severity.WATCH
        else:
            severity = Severity.NORMAL

        # Anomaly override: escalate to ALARM if score breaches limit
        if anomaly_score < thresholds.anomaly_score_limit and severity < Severity.ALARM:
            severity = Severity.ALARM

        alert = Alert(
            equipment_id=equipment_id,
            severity=severity,
            rul_hours=rul_hours,
            anomaly_score=anomaly_score,
            fault_label=fault_label,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message=f"{equipment_id}: {severity.name} -- RUL {rul_hours:.0f}h, {fault_label}",
        )
        return alert

    def send(self, alert: Alert) -> None:
        """Push alert to AEGIS-MONITOR WebSocket gateway."""
        payload = json.dumps(asdict(alert))
        try:
            with ws_client.connect(self._ws_url) as conn:
                conn.send(payload)
                logger.info("Alert dispatched: %s", alert.message)
        except Exception:
            logger.exception("Failed to dispatch alert to %s", self._ws_url)
