"""Tests for Settings.from_yaml()."""

from __future__ import annotations

from pathlib import Path

from triton_ml.config import Settings

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestFromYaml:
    """The shipped example config must load into a valid Settings object."""

    def test_example_config_loads(self) -> None:
        settings = Settings.from_yaml(REPO_ROOT / "config.example.yaml")
        assert settings.model.xgb_n_estimators == 400
        assert settings.alerts.shutdown_rul_hours == 24.0
        assert settings.paths.onnx_export == Path("models/onnx")
        assert settings.onnx.dynamic_axes == {"input": {0: "batch"}, "output": {0: "batch"}}

    def test_missing_sections_fall_back_to_defaults(self, tmp_path: Path) -> None:
        cfg = tmp_path / "partial.yaml"
        cfg.write_text("alerts:\n  shutdown_rul_hours: 8.0\n", encoding="utf-8")
        settings = Settings.from_yaml(cfg)
        assert settings.alerts.shutdown_rul_hours == 8.0
        assert settings.alerts.watch_rul_hours == Settings().alerts.watch_rul_hours
        assert settings.model == Settings().model

    def test_empty_file_yields_defaults(self, tmp_path: Path) -> None:
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("", encoding="utf-8")
        assert Settings.from_yaml(cfg) == Settings()

    def test_paths_are_converted_to_path_objects(self, tmp_path: Path) -> None:
        cfg = tmp_path / "paths.yaml"
        cfg.write_text('paths:\n  raw_telemetry: "/mnt/telemetry"\n', encoding="utf-8")
        settings = Settings.from_yaml(cfg)
        assert settings.paths.raw_telemetry == Path("/mnt/telemetry")
