"""
Vibration feature extraction from hull and machinery accelerometers.

Follows ISO 10816 / ISO 20816 vibration severity classification
for rotating marine machinery (main engines, turbochargers, pumps).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from dataclasses import dataclass


@dataclass
class VibrationFeatures:
    """Container for a single vibration analysis window."""

    rms: float
    kurtosis: float
    crest_factor: float
    dominant_freq_hz: float
    spectral_energy: NDArray[np.float64]
    envelope_peak: float


class VibrationFeatureExtractor:
    """Extract condition indicators from raw accelerometer time-series.

    Designed for sampling rates typical of marine CBM systems (2-25 kHz).
    """

    def __init__(self, sampling_rate_hz: int = 10_000, window_size: int = 4096) -> None:
        self._fs = sampling_rate_hz
        self._win = window_size

    def extract(self, raw: NDArray[np.float64]) -> VibrationFeatures:
        """Compute vibration features for a single analysis window."""
        rms = float(np.sqrt(np.mean(raw ** 2)))
        kurt = float(self._kurtosis(raw))
        crest = float(np.max(np.abs(raw)) / rms) if rms > 1e-9 else 0.0

        freqs, psd = signal.welch(raw, fs=self._fs, nperseg=self._win)
        dominant_idx = int(np.argmax(psd))
        dominant_freq = float(freqs[dominant_idx])

        # Envelope analysis via Hilbert transform -- detects bearing defects
        analytic = signal.hilbert(raw)
        envelope = np.abs(analytic)
        envelope_peak = float(np.max(envelope))

        return VibrationFeatures(
            rms=rms,
            kurtosis=kurt,
            crest_factor=crest,
            dominant_freq_hz=dominant_freq,
            spectral_energy=psd,
            envelope_peak=envelope_peak,
        )

    @staticmethod
    def _kurtosis(x: NDArray[np.float64]) -> float:
        """Excess kurtosis -- elevated values signal impulsive bearing faults."""
        n = len(x)
        mean = np.mean(x)
        m4 = np.sum((x - mean) ** 4) / n
        m2 = np.sum((x - mean) ** 2) / n
        return float(m4 / (m2 ** 2) - 3.0) if m2 > 1e-12 else 0.0
