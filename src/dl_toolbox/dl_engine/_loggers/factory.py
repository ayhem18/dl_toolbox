"""Build a ``Logger`` by backend name. The caller builds the manager."""

from __future__ import annotations

from dl_toolbox.dl_engine._loggers.base_logger import Logger


def get_logger(name: str, *, manager, **kwargs) -> Logger:
    if name == "tensorboard":
        from dl_toolbox.dl_engine._loggers.tensorboard.tensorboard_logger import TensorBoardLogger

        return TensorBoardLogger(manager=manager, **kwargs)
    if name == "wandb":
        from dl_toolbox.dl_engine._loggers.wandb.wandb_logger import WandbLogger

        return WandbLogger(manager=manager, **kwargs)
    if name == "mlflow":
        from dl_toolbox.dl_engine._loggers.mlflow.mlflow_logger import MlflowLogger

        return MlflowLogger(manager=manager, **kwargs)
    raise ValueError(f"Unknown logger: {name}")
