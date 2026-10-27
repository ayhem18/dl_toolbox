from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainerConfig:
    max_epochs: int
    accelerator: str
    devices: int | str
    precision: int | str
    enable_progress_bar: bool


@dataclass(frozen=True)
class MonitorConfig:
    key: str
    mode: str


@dataclass(frozen=True)
class EarlyStoppingConfig:
    NONE = "none"
    PATIENCE = "patience"

    name: str
    patience: int | None = None
    min_delta: float | None = None

    @classmethod
    def none(cls) -> EarlyStoppingConfig:
        return cls(name=cls.NONE)


@dataclass(frozen=True)
class SeedConfig:
    seed: int


@dataclass(frozen=True)
class FitConfig:
    trainer: TrainerConfig
    monitor: MonitorConfig
    early_stopping: EarlyStoppingConfig
