"""
ONNX model exporter for shipboard edge deployment.

Exports trained PyTorch RUL models to ONNX format for inference
on Advantech UNO-2484G marine-grade edge computers or similar
IEC 61850-compliant industrial PCs running ONNX Runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import onnx
import onnxruntime as ort
import numpy as np

from triton_ml.config import Settings
from triton_ml.models.rul_estimator import RULEstimator

logger = logging.getLogger(__name__)


class ONNXExporter:
    """Export PyTorch models to optimised ONNX graphs."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._cfg = settings or Settings()

    def export_rul(self, estimator: RULEstimator, input_dim: int,
                   dest: Optional[Path] = None) -> Path:
        """Export the RUL network to ONNX format.

        Args:
            estimator: trained RULEstimator instance.
            input_dim: number of input features.
            dest: output file path; defaults to configured export directory.
        """
        out_path = dest or self._cfg.paths.onnx_export / "rul_model.onnx"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        model = estimator._model
        model.eval()
        dummy = torch.randn(1, input_dim)

        torch.onnx.export(
            model, dummy, str(out_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=self._cfg.onnx.dynamic_axes,
            opset_version=self._cfg.onnx.opset_version,
        )

        # Validate exported graph
        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model exported and validated: %s", out_path)
        return out_path

    def verify(self, onnx_path: Path, sample_input: np.ndarray) -> np.ndarray:
        """Run inference via ONNX Runtime to verify export correctness."""
        session = ort.InferenceSession(str(onnx_path))
        result = session.run(
            None, {"input": sample_input.astype(np.float32)}
        )
        return result[0]
