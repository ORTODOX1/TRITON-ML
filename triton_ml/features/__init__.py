"""
Unified feature pipeline combining vibration, thermal, and operational extractors.

Orchestrates parallel extraction and outputs a flat feature vector
suitable for downstream fault classification and RUL estimation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import asdict
from typing import Dict, Any

from triton_ml.features.vibration import VibrationFeatureExtractor
from triton_ml.features.thermal import ThermalFeatureExtractor
from triton_ml.features.operational import OperationalFeatureExtractor


class FeaturePipeline:
    """Combine all domain extractors into a single feature vector.

    Each extractor is independently configurable; the pipeline
    merges their outputs into a dict keyed by `<domain>__<feature>`.
    """

    def __init__(
        self,
        vibration: VibrationFeatureExtractor | None = None,
        thermal: ThermalFeatureExtractor | None = None,
        operational: OperationalFeatureExtractor | None = None,
    ) -> None:
        self.vibration = vibration or VibrationFeatureExtractor()
        self.thermal = thermal or ThermalFeatureExtractor()
        self.operational = operational or OperationalFeatureExtractor()

    def run(
        self,
        vib_raw: NDArray[np.float64],
        thermal_readings: NDArray[np.float64],
        rpm: NDArray[np.float64],
        fuel_flow: NDArray[np.float64],
        torque: NDArray[np.float64],
        scav_pressure: NDArray[np.float64],
    ) -> Dict[str, Any]:
        """Execute all extractors and return merged feature dictionary."""
        vib = self.vibration.extract(vib_raw)
        therm = self.thermal.extract(thermal_readings)
        ops = self.operational.extract(rpm, fuel_flow, torque, scav_pressure)

        features: Dict[str, Any] = {}
        for prefix, feat_obj in [("vib", vib), ("therm", therm), ("ops", ops)]:
            for key, value in asdict(feat_obj).items():
                if isinstance(value, np.ndarray):
                    continue  # skip raw spectral arrays
                features[f"{prefix}__{key}"] = value
        return features
