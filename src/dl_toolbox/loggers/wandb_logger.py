from typing import Any, Dict

from .base import BaseLogger

_EXTRA = "wandb"


class WandbLogger(BaseLogger):
    @staticmethod
    def _import_wandb():
        return WandbLogger._import_package("wandb", extra=_EXTRA)

    def __init__(self, log_dir: str, **kwargs):
        wandb = self._import_wandb()
        super().__init__(log_dir, **kwargs)
        project = kwargs.get("project")
        run_name = kwargs.get("run_name", kwargs.get("name"))
        init_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"project", "run_name", "name"}
        }
        wandb.init(dir=str(self.log_dir), project=project, name=run_name, **init_kwargs)

    def log_scalar(self, step: int, tag: str, value: float):
        wandb = self._import_wandb()
        wandb.log({tag: value}, step=step)

    def log_dict(self, step: int, values: Dict[str, float]):
        wandb = self._import_wandb()
        wandb.log(values, step=step)

    def log_image(self, step: int, tag: str, image: Any, **kwargs):
        wandb = self._import_wandb()
        wandb.log({tag: wandb.Image(image, **kwargs)}, step=step)

    def log_table(self, step: int, tag: str, data: Any, **kwargs):
        wandb = self._import_wandb()
        wandb.log({tag: wandb.Table(data=data, **kwargs)}, step=step)

    def log_histogram(self, step: int, tag: str, values: Any):
        wandb = self._import_wandb()
        wandb.log({tag: wandb.Histogram(values)}, step=step)

    def log_config(self, config: Dict, config_file_name: str):
        wandb = self._import_wandb()
        super().log_config(config, config_file_name)
        wandb.config.update(config)

    def close(self):
        wandb = self._import_wandb()
        wandb.finish()
