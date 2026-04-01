"""
TRITON-ML model zoo.

Fault classification (XGBoost), remaining useful life estimation
(PyTorch DNN with MC-dropout), and anomaly detection (Isolation Forest).
"""

from triton_ml.models.fault_classifier import FaultClassifier
from triton_ml.models.rul_estimator import RULEstimator
from triton_ml.models.anomaly import AnomalyDetector

__all__ = ["FaultClassifier", "RULEstimator", "AnomalyDetector"]
