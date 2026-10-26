from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dl_toolbox.loggers.base_logger import Logger


@dataclass(frozen=True)
class TrainerConfig:
    max_epochs: int
    accelerator: str
    devices: int | str
    enable_progress_bar: bool


@dataclass(frozen=True)
class MonitorConfig:
    key: str
    mode: str


@dataclass(frozen=True)
class CheckpointConfig:
    dirpath: str | Path
    filename: str
    save_top_k: int
    save_last: bool


@dataclass(frozen=True)
class EarlyStoppingConfig:
    patience: int
    min_delta: float


@dataclass(frozen=True)
class LoggerConfig:
    logger: Logger
    run_config: dict[str, Any]
    run_config_filename: str


@dataclass(frozen=True)
class FitConfig:
    trainer: TrainerConfig
    monitor: MonitorConfig
    checkpoint: CheckpointConfig
    logger: LoggerConfig
    early_stopping: EarlyStoppingConfig | None
