"""Configuration loading.

Configs are plain YAML; we wrap them in light dataclasses so downstream code gets
attribute access and a single place to document intent. Anything not explicitly
modeled here stays reachable via the raw dict on ``.raw``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class SensorConfig:
    raw: dict[str, Any]

    @property
    def sample_rates(self) -> dict[str, float]:
        return self.raw["sample_rates_hz"]

    @property
    def classes(self) -> list[str]:
        return self.raw["classes"]

    def strain_channel_names(self) -> list[str]:
        s = self.raw["strain"]
        return [c["name"] for c in s["bend"]] + [c["name"] for c in s["stretch"]]

    def mic_channel_names(self) -> list[str]:
        """The frozen model contract: every mic slot, working or not."""
        return list(self.raw["mic"]["channels"])

    def mic_active_channels(self) -> list[str]:
        """Mic channels actually read from disk; the rest are zero-filled.

        Defaults to every channel, so a config without the key behaves as before.
        """
        return list(self.raw["mic"].get("active_channels", self.raw["mic"]["channels"]))

    def imu_channel_names(self) -> list[str]:
        imu = self.raw["imu"]
        return list(imu["accel_channels"]) + list(imu["gyro_channels"])

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SensorConfig":
        return cls(_load_yaml(path or CONFIG_DIR / "sensors.yaml"))


@dataclass
class PipelineConfig:
    raw: dict[str, Any]

    @property
    def target_grid_hz(self) -> float:
        return float(self.raw["resampling"]["target_grid_hz"])

    @property
    def window_s(self) -> float:
        return float(self.raw["windowing"]["window_s"])

    @property
    def overlap(self) -> float:
        return float(self.raw["windowing"]["overlap"])

    @property
    def hop_s(self) -> float:
        return self.window_s * (1.0 - self.overlap)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PipelineConfig":
        return cls(_load_yaml(path or CONFIG_DIR / "pipeline.yaml"))


@dataclass
class ModelConfig:
    raw: dict[str, Any]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ModelConfig":
        return cls(_load_yaml(path or CONFIG_DIR / "model.yaml"))


@dataclass
class Configs:
    sensors: SensorConfig = field(default_factory=SensorConfig.load)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig.load)
    model: ModelConfig = field(default_factory=ModelConfig.load)

    @classmethod
    def load(cls) -> "Configs":
        return cls()
