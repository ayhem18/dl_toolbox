from typing import Any, Dict

from .base import BaseLogger


class PlaceholderLogger(BaseLogger):
    def __init__(self, log_dir: str, **kwargs):
        super().__init__(log_dir, **kwargs)

    def log_scalar(self, step: int, tag: str, value: float):
        pass

    def log_dict(self, step: int, values: Dict[str, float]):
        pass

    def log_image(self, step: int, tag: str, image: Any, **kwargs):
        pass

    def log_table(self, step: int, tag: str, data: Any, **kwargs):
        pass

    def log_histogram(self, step: int, tag: str, values: Any):
        pass

    def log_config(self, config: Dict, config_file_name: str):
        super().log_config(config, config_file_name)

    def save(self, object: Any, file_name: str):
        super().save(object, file_name)

    def close(self):
        pass
