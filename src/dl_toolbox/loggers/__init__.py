from .base import BaseLogger


def get_logger(name: str, **kwargs) -> BaseLogger:
    if name == "tensorboard":
        from .tensorboard_logger import TensorBoardLogger

        return TensorBoardLogger(**kwargs)
    if name == "wandb":
        from .wandb_logger import WandbLogger

        return WandbLogger(**kwargs)
    if name is None or name == "placeholder":
        from .placeholder_logger import PlaceholderLogger

        return PlaceholderLogger(**kwargs)
    raise ValueError(f"Unknown logger: {name}")
