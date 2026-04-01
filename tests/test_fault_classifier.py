"""Tests for FaultClassifier."""

from __future__ import annotations

import numpy as np
import pytest

from triton_ml.models.fault_classifier import FaultClassifier
from triton_ml.config import Settings


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic multi-class classification data."""
    rng = np.random.default_rng(42)
    n_samples, n_features, n_classes = 200, 10, 6
    X = rng.standard_normal((n_samples, n_features))
    y = rng.integers(0, n_classes, size=n_samples).astype(np.int64)
    return X, y


@pytest.fixture
def trained_classifier(synthetic_data: tuple[np.ndarray, np.ndarray]) -> FaultClassifier:
    """Return a classifier trained on synthetic data."""
    X, y = synthetic_data
    clf = FaultClassifier(settings=Settings())
    clf.train(X, y)
    return clf


class TestTraining:
    """Verify the classifier trains without errors on synthetic data."""

    def test_train_completes(self, synthetic_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic_data
        clf = FaultClassifier()
        clf.train(X, y)
        # After training, explainer should be initialized
        assert clf._explainer is not None

    def test_train_with_custom_settings(self, synthetic_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic_data
        settings = Settings()
        clf = FaultClassifier(settings=settings)
        clf.train(X, y)
        assert clf._explainer is not None


class TestPrediction:
    """Prediction output shape and value range checks."""

    def test_predict_shape(
        self, trained_classifier: FaultClassifier, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, _ = synthetic_data
        preds = trained_classifier.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_predict_valid_classes(
        self, trained_classifier: FaultClassifier, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, _ = synthetic_data
        preds = trained_classifier.predict(X)
        assert all(p in FaultClassifier.FAULT_LABELS for p in preds)

    def test_predict_proba_shape(
        self, trained_classifier: FaultClassifier, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, _ = synthetic_data
        proba = trained_classifier.predict_proba(X)
        assert proba.shape == (X.shape[0], len(FaultClassifier.FAULT_LABELS))

    def test_predict_proba_sums_to_one(
        self, trained_classifier: FaultClassifier, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, _ = synthetic_data
        proba = trained_classifier.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


class TestExplainability:
    """SHAP value generation tests."""

    def test_explain_returns_shap_values(
        self, trained_classifier: FaultClassifier, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, _ = synthetic_data
        explanation = trained_classifier.explain(X[:5])
        assert explanation.values is not None

    def test_explain_before_training_raises(self) -> None:
        clf = FaultClassifier()
        X = np.random.default_rng(0).standard_normal((5, 10))
        with pytest.raises(RuntimeError, match="trained"):
            clf.explain(X)
