"""
Operational feature extraction from engine performance telemetry.

Computes load factor, specific fuel oil consumption (SFOC), and
power output metrics used for performance-based fault detection
in slow-speed two-stroke and medium-speed four-stroke marine diesels.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass
class OperationalFeatures:
    """Engine performance indicators for a telemetry window."""

    load_factor: float          # fraction of MCR (0..1)
    sfoc_g_per_kwh: float       # specific fuel oil consumption
    power_output_kw: float
    scavenge_pressure_bar: float
    torque_deviation_pct: float


class OperationalFeatureExtractor:
    """Derive engine health signals from performance telemetry.

    MCR = Maximum Continuous Rating as per engine shop test protocol.
    """

    def __init__(self, mcr_kw: float = 12_000.0) -> None:
        if mcr_kw <= 0:
            raise ValueError("MCR must be a positive value in kW")
        self._mcr_kw = mcr_kw

    def extract(
        self,
        rpm: NDArray[np.float64],
        fuel_flow_kg_h: NDArray[np.float64],
        torque_nm: NDArray[np.float64],
        scav_pressure_bar: NDArray[np.float64],
    ) -> OperationalFeatures:
        """Compute operational features from aligned telemetry channels.

        All input arrays must share the same time axis length.
        """
        # Instantaneous shaft power P = 2 * pi * n * T / 60
        power = 2.0 * np.pi * rpm * torque_nm / 60.0
        mean_power_kw = float(np.mean(power) / 1_000.0)

        load_factor = mean_power_kw / self._mcr_kw
        load_factor = float(np.clip(load_factor, 0.0, 1.2))  # allow slight overload

        # SFOC: grams of fuel per kWh produced -- key degradation metric
        mean_fuel = float(np.mean(fuel_flow_kg_h)) * 1_000.0  # kg/h -> g/h
        sfoc = mean_fuel / mean_power_kw if mean_power_kw > 1.0 else 0.0

        # Torque deviation from running mean indicates cylinder imbalance
        running_mean = np.convolve(torque_nm, np.ones(10) / 10, mode="same")
        deviation = np.abs(torque_nm - running_mean) / (running_mean + 1e-6)
        torque_dev_pct = float(np.mean(deviation) * 100.0)

        return OperationalFeatures(
            load_factor=load_factor,
            sfoc_g_per_kwh=sfoc,
            power_output_kw=mean_power_kw,
            scavenge_pressure_bar=float(np.mean(scav_pressure_bar)),
            torque_deviation_pct=torque_dev_pct,
        )
