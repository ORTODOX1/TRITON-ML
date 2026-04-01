"""
Thermal feature extraction from engine and exhaust temperature arrays.

Marine diesel exhaust-gas temperature deviation is a primary indicator
of cylinder imbalance, turbocharger fouling, and injector degradation
(ref: MAN Energy Solutions Technical Paper 2019-01).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List


@dataclass
class ThermalFeatures:
    """Thermal condition indicators for a sensor array snapshot."""

    mean_temp_c: float
    max_delta_t: float
    gradient_c_per_min: float
    trend_slope: float
    cylinder_spread: float


class ThermalFeatureExtractor:
    """Derive degradation signals from temperature sensor arrays.

    Expects arrays shaped (n_timestamps, n_sensors) sampled at 1 Hz.
    """

    def __init__(self, sensor_count: int = 12, sample_interval_s: float = 1.0) -> None:
        self._n_sensors = sensor_count
        self._dt = sample_interval_s

    def extract(self, readings: NDArray[np.float64]) -> ThermalFeatures:
        """Compute thermal features over a time window.

        Args:
            readings: shape (n_timestamps, n_sensors) in degrees Celsius.
        """
        mean_temp = float(np.mean(readings))

        # Max inter-cylinder temperature deviation at each timestep
        spreads = np.max(readings, axis=1) - np.min(readings, axis=1)
        max_delta_t = float(np.max(spreads))
        cylinder_spread = float(np.mean(spreads))

        # Temporal gradient: rate of temperature change (deg C / min)
        mean_per_step = np.mean(readings, axis=1)
        gradient = float(np.mean(np.diff(mean_per_step) / self._dt) * 60.0)

        # Linear trend slope via least-squares -- rising slope warns of fouling
        trend_slope = self._linear_slope(mean_per_step)

        return ThermalFeatures(
            mean_temp_c=mean_temp,
            max_delta_t=max_delta_t,
            gradient_c_per_min=gradient,
            trend_slope=trend_slope,
            cylinder_spread=cylinder_spread,
        )

    def extract_batch(self, windows: List[NDArray[np.float64]]) -> List[ThermalFeatures]:
        """Extract features from multiple consecutive windows."""
        return [self.extract(w) for w in windows]

    @staticmethod
    def _linear_slope(series: NDArray[np.float64]) -> float:
        """Ordinary least-squares slope for a 1-D time-series."""
        n = len(series)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        denominator = n * np.dot(x, x) - x.sum() ** 2
        if abs(denominator) < 1e-12:
            return 0.0
        slope = float((n * np.dot(x, series) - x.sum() * series.sum()) / denominator)
        return slope
