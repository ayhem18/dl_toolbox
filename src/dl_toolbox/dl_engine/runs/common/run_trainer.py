"""Lightning Trainer mapping shared by train/val and train-only. Not GPU picking."""

from __future__ import annotations

import lightning as L

from dl_toolbox.dl_engine.runs.common.run_config import Accelerator, Strategy, TrainerConfig


class FitError(ValueError):
    pass


def lightning_devices(
    *, accelerator: str, devices: tuple[int, ...]
) -> int | list[int]:
    if accelerator == Accelerator.CPU:
        return 1
    return list(devices)


def lightning_strategy(strategy: str) -> str:
    if strategy == Strategy.DDP:
        return "ddp"
    return "auto"


def require_single_machine(*, trainer: L.Trainer, config: TrainerConfig) -> None:
    if int(trainer.num_nodes) > 1:
        raise FitError(
            f"this run is single-machine; num_nodes={trainer.num_nodes} is illegal"
        )
    world_size = int(trainer.world_size)
    if config.strategy == Strategy.DDP:
        if world_size != len(config.devices):
            raise FitError(
                "ddp world_size must equal len(devices), "
                f"got world_size={world_size} devices={config.devices!r}"
            )
        return
    if world_size != 1:
        raise FitError(f"single_device world_size must be 1, got {world_size}")
