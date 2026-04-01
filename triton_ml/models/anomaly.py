"""
Anomaly detection via Isolation Forest for novel failure modes.

Catches out-of-distribution machinery behaviour that the supervised
fault classifier has never seen -- critical for newly commissioned
vessels or after major overhaul when baseline patterns shift.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Optional
import pickle

from sklearn.ensemble import IsolationForest

from triton_ml.config import Settings


class AnomalyDetector:
    """Isolation Forest wrapper tuned for marine telemetry streams."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._cfg = settings or Settings()
        self._model = IsolationForest(
            contamination=self._cfg.model.isolation_contamination,
            n_estimators=200,
            max_samples="auto",
            random_state=42,
        )
        self._fitted = False

    def fit(self, X: NDArray[np.float64]) -> None:
        """Learn normal operating envelope from healthy baseline data."""
        self._model.fit(X)
        self._fitted = True

    def score(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return anomaly scores; more negative = more anomalous."""
        self._ensure_fitted()
        return self._model.score_samples(X)

    def is_anomalous(self, X: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Binary anomaly flag per sample (True = anomaly)."""
        scores = self.score(X)
        return scores < self._cfg.alerts.anomaly_score_limit

    def save(self, path: Optional[Path] = None) -> Path:
        """Serialize fitted model."""
        dest = path or self._cfg.paths.trained_models / "anomaly_detector.pkl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            pickle.dump(self._model, f)
        return dest

    def load(self, path: Path) -> None:
        """Load previously fitted model."""
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        self._fitted = True

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("AnomalyDetector must be fitted before scoring")
