"""
TRITON-ML: Predictive Maintenance ML for Ship Machinery.

Provides fault classification, remaining useful life estimation,
and anomaly detection for marine propulsion and auxiliary systems.
Designed for integration with AEGIS-MONITOR telemetry pipeline.
"""

__version__ = "0.4.0"
__author__ = "Fincantieri Digital"

from triton_ml.config import Settings
from triton_ml.alerting import AlertEngine
from triton_ml.export import ONNXExporter

__all__ = [
    "Settings",
    "AlertEngine",
    "ONNXExporter",
]
