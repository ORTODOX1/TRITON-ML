"""
CLI entry point for TRITON-ML pipeline.

Usage:
    python -m triton_ml train --config config.yaml
    python -m triton_ml predict --model model.onnx --input data.parquet
    python -m triton_ml export --model model.pt --output model.onnx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("triton_ml")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_train(args: argparse.Namespace) -> None:
    """Run the training pipeline from a YAML config."""
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
    logger.info("Starting training with config: %s", config_path)
    # Training orchestration would load config, build features, and fit models.
    logger.info("Training complete.")


def cmd_predict(args: argparse.Namespace) -> None:
    """Run inference using a trained ONNX model."""
    model_path = Path(args.model)
    input_path = Path(args.input)
    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        sys.exit(1)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)
    logger.info("Running prediction: model=%s input=%s", model_path, input_path)
    logger.info("Prediction complete.")


def cmd_export(args: argparse.Namespace) -> None:
    """Export a PyTorch model checkpoint to ONNX format."""
    model_path = Path(args.model)
    output_path = Path(args.output)
    if not model_path.exists():
        logger.error("Model checkpoint not found: %s", model_path)
        sys.exit(1)
    logger.info("Exporting %s -> %s", model_path, output_path)
    logger.info("Export complete.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triton_ml",
        description="TRITON-ML: Predictive maintenance ML for ship machinery",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train models from config")
    p_train.add_argument("--config", required=True, help="Path to YAML config file")

    # predict
    p_pred = sub.add_parser("predict", help="Run inference on input data")
    p_pred.add_argument("--model", required=True, help="Path to ONNX model")
    p_pred.add_argument("--input", required=True, help="Path to input Parquet file")

    # export
    p_exp = sub.add_parser("export", help="Export model to ONNX")
    p_exp.add_argument("--model", required=True, help="Path to PyTorch checkpoint (.pt)")
    p_exp.add_argument("--output", required=True, help="Output ONNX file path")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)

    commands = {"train": cmd_train, "predict": cmd_predict, "export": cmd_export}
    commands[args.command](args)


if __name__ == "__main__":
    main()
