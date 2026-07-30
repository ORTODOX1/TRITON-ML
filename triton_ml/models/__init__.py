"""
TRITON-ML model zoo.

Fault classification (XGBoost), remaining useful life estimation
(PyTorch DNN with MC-dropout), and anomaly detection (Isolation Forest).
"""

from triton_ml.models.anomaly import AnomalyDetector
from triton_ml.models.fault_classifier import FaultClassifier
from triton_ml.models.rul_estimator import RULEstimator

__all__ = ["AnomalyDetector", "FaultClassifier", "RULEstimator"]
