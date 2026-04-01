"""Tests for VibrationFeatureExtractor."""

from __future__ import annotations

import numpy as np
import pytest

from triton_ml.features.vibration import VibrationFeatureExtractor


@pytest.fixture
def extractor() -> VibrationFeatureExtractor:
    return VibrationFeatureExtractor(sampling_rate_hz=10_000, window_size=1024)


def _make_sine(freq_hz: float, fs: int, duration_s: float) -> np.ndarray:
    """Generate a pure sinusoidal signal at the given frequency."""
    t = np.arange(0, duration_s, 1.0 / fs)
    return np.sin(2.0 * np.pi * freq_hz * t)


class TestFFTDominantFrequency:
    """Welch PSD should identify the dominant frequency of a pure tone."""

    def test_single_tone_100hz(self, extractor: VibrationFeatureExtractor) -> None:
        sig = _make_sine(freq_hz=100.0, fs=10_000, duration_s=0.5)
        features = extractor.extract(sig)
        assert abs(features.dominant_freq_hz - 100.0) < 20.0

    def test_dominant_among_two_tones(self, extractor: VibrationFeatureExtractor) -> None:
        """Stronger tone should be reported as dominant frequency."""
        t = np.arange(0, 0.5, 1.0 / 10_000)
        sig = 3.0 * np.sin(2.0 * np.pi * 200.0 * t) + 0.5 * np.sin(2.0 * np.pi * 800.0 * t)
        features = extractor.extract(sig)
        assert abs(features.dominant_freq_hz - 200.0) < 20.0


class TestRMS:
    """RMS of a sine wave with amplitude A should be A / sqrt(2)."""

    def test_unit_amplitude(self, extractor: VibrationFeatureExtractor) -> None:
        sig = _make_sine(freq_hz=50.0, fs=10_000, duration_s=0.5)
        features = extractor.extract(sig)
        expected_rms = 1.0 / np.sqrt(2.0)
        assert abs(features.rms - expected_rms) < 0.01

    def test_zero_signal_rms(self, extractor: VibrationFeatureExtractor) -> None:
        sig = np.zeros(4096)
        features = extractor.extract(sig)
        assert features.rms == pytest.approx(0.0, abs=1e-9)


class TestKurtosis:
    """Excess kurtosis: ~0 for Gaussian, elevated for impulsive signals."""

    def test_gaussian_near_zero(self) -> None:
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(50_000)
        kurt = VibrationFeatureExtractor._kurtosis(sig)
        assert abs(kurt) < 0.2

    def test_impulsive_elevated(self) -> None:
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(10_000)
        # Inject periodic impulses to simulate bearing defect
        sig[::200] += 15.0
        kurt = VibrationFeatureExtractor._kurtosis(sig)
        assert kurt > 3.0


class TestEnvelopePeak:
    """Envelope peak should scale with signal amplitude."""

    def test_envelope_scales_with_amplitude(self, extractor: VibrationFeatureExtractor) -> None:
        sig_low = _make_sine(freq_hz=100.0, fs=10_000, duration_s=0.5) * 1.0
        sig_high = _make_sine(freq_hz=100.0, fs=10_000, duration_s=0.5) * 5.0
        feat_low = extractor.extract(sig_low)
        feat_high = extractor.extract(sig_high)
        assert feat_high.envelope_peak > feat_low.envelope_peak * 3.0
