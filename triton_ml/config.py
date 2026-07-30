"""
Central configuration for TRITON-ML pipeline.

Thresholds are project defaults chosen for demonstration purposes.
They are not derived from any standard or certified dataset and must
be tuned per vessel and equipment class before any practical use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class DataPaths:
    """Filesystem layout for raw telemetry and trained artefacts."""

    raw_telemetry: Path = Path("data/raw")
    processed_features: Path = Path("data/features")
    trained_models: Path = Path("models/checkpoints")
    onnx_export: Path = Path("models/onnx")


@dataclass(frozen=True)
class ModelParams:
    """Hyper-parameters shared across all model stages."""

    xgb_n_estimators: int = 400
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    rul_hidden_dim: int = 128
    rul_dropout: float = 0.20
    rul_mc_samples: int = 50
    isolation_contamination: float = 0.02


@dataclass(frozen=True)
class AlertThresholds:
    """Four-tier alert boundaries expressed in remaining-useful-life hours."""

    watch_rul_hours: float = 720.0   # 30 days
    alarm_rul_hours: float = 168.0   # 7 days
    shutdown_rul_hours: float = 24.0
    anomaly_score_limit: float = -0.35


@dataclass(frozen=True)
class ONNXConfig:
    """Settings for ONNX export targeting edge inference on shipboard hardware."""

    opset_version: int = 17
    dynamic_axes: Dict[str, Dict[int, str]] = field(
        default_factory=lambda: {"input": {0: "batch"}, "output": {0: "batch"}}
    )


@dataclass(frozen=True)
class Settings:
    """Top-level configuration container."""

    paths: DataPaths = DataPaths()
    model: ModelParams = ModelParams()
    alerts: AlertThresholds = AlertThresholds()
    onnx: ONNXConfig = ONNXConfig()

    @classmethod
    def from_yaml(cls, path: Path | str) -> Settings:
        """Build a Settings instance from a YAML file.

        Recognised top-level sections are ``paths``, ``model``, ``alerts``
        and ``onnx``. Any section that is absent falls back to the defaults
        defined above; see ``config.example.yaml`` for the full layout.
        """
        with open(path, "r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}

        paths = {key: Path(value) for key, value in raw.get("paths", {}).items()}
        return cls(
            paths=DataPaths(**paths),
            model=ModelParams(**raw.get("model", {})),
            alerts=AlertThresholds(**raw.get("alerts", {})),
            onnx=ONNXConfig(**raw.get("onnx", {})),
        )
