"""
XGBoost-based fault classifier for marine machinery.

Classifies operating condition into fault categories defined by
the vessel's planned maintenance system (PMS):
  0 = NORMAL, 1 = BEARING_WEAR, 2 = MISALIGNMENT,
  3 = IMBALANCE, 4 = FOULING, 5 = INJECTOR_FAULT.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Dict, Optional

import xgboost as xgb
import shap

from triton_ml.config import Settings


class FaultClassifier:
    """Gradient-boosted fault classifier with SHAP explainability."""

    FAULT_LABELS = {
        0: "NORMAL", 1: "BEARING_WEAR", 2: "MISALIGNMENT",
        3: "IMBALANCE", 4: "FOULING", 5: "INJECTOR_FAULT",
    }

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._cfg = settings or Settings()
        self._model = xgb.XGBClassifier(
            n_estimators=self._cfg.model.xgb_n_estimators,
            max_depth=self._cfg.model.xgb_max_depth,
            learning_rate=self._cfg.model.xgb_learning_rate,
            objective="multi:softprob",
            num_class=len(self.FAULT_LABELS),
            tree_method="hist",
            eval_metric="mlogloss",
        )
        self._explainer: Optional[shap.TreeExplainer] = None

    def train(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> None:
        """Fit classifier on labelled maintenance records."""
        self._model.fit(X, y)
        self._explainer = shap.TreeExplainer(self._model)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Return predicted fault class indices."""
        return self._model.predict(X)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return per-class probabilities for alert prioritisation."""
        return self._model.predict_proba(X)

    def explain(self, X: NDArray[np.float64]) -> shap.Explanation:
        """Generate SHAP values for root-cause analysis reports."""
        if self._explainer is None:
            raise RuntimeError("Model must be trained before explanation")
        return self._explainer(X)

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist model to JSON for reproducibility."""
        dest = path or self._cfg.paths.trained_models / "fault_classifier.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(dest))
        return dest

    def load(self, path: Path) -> None:
        """Load a previously trained model."""
        self._model.load_model(str(path))
        self._explainer = shap.TreeExplainer(self._model)
