"""
CLI entry point for TRITON-ML pipeline.

Usage:
    python -m triton_ml train --config config.yaml
    python -m triton_ml predict --model model.onnx --input data.csv
    python -m triton_ml export --model model.pt --output model.onnx

Scope note: these subcommands currently validate their arguments and load
configuration. The training, inference and export orchestration is not
wired up yet -- use the library API (triton_ml.models, triton_ml.export)
directly in the meantime.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from triton_ml.config import Settings

logger = logging.getLogger("triton_ml")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_train(args: argparse.Namespace) -> None:
    """Validate the training configuration.

    Loads and validates the YAML config. The training loop itself is not
    implemented yet -- see the module docstring.
    """
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
    settings = Settings.from_yaml(config_path)
    logger.info("Config loaded: %s", config_path)
    logger.info(
        "Model params: xgb_n_estimators=%d rul_hidden_dim=%d rul_mc_samples=%d",
        settings.model.xgb_n_estimators,
        settings.model.rul_hidden_dim,
        settings.model.rul_mc_samples,
    )
    logger.warning("Training orchestration is not implemented yet; nothing was fitted.")


def cmd_predict(args: argparse.Namespace) -> None:
    """Validate the inference arguments.

    The inference loop is not implemented yet -- see the module docstring.
    """
    model_path = Path(args.model)
    input_path = Path(args.input)
    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        sys.exit(1)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)
    logger.info("Inputs validated: model=%s input=%s", model_path, input_path)
    logger.warning("Inference orchestration is not implemented yet; nothing was predicted.")


def cmd_export(args: argparse.Namespace) -> None:
    """Validate the export arguments.

    Use triton_ml.export.ONNXExporter for the actual conversion.
    """
    model_path = Path(args.model)
    output_path = Path(args.output)
    if not model_path.exists():
        logger.error("Model checkpoint not found: %s", model_path)
        sys.exit(1)
    logger.info("Inputs validated: %s -> %s", model_path, output_path)
    logger.warning("Export orchestration is not implemented yet; use ONNXExporter directly.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triton_ml",
        description="TRITON-ML: Predictive maintenance ML for ship machinery",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Load and validate a training config")
    p_train.add_argument("--config", required=True, help="Path to YAML config file")

    # predict
    p_pred = sub.add_parser("predict", help="Validate inference inputs")
    p_pred.add_argument("--model", required=True, help="Path to ONNX model")
    p_pred.add_argument("--input", required=True, help="Path to input data file")

    # export
    p_exp = sub.add_parser("export", help="Validate ONNX export inputs")
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
