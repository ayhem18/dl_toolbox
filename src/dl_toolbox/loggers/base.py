import importlib
import json
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict

from dl_toolbox.paths import process_path


class BaseLogger(ABC):
    """Abstract logger. Concrete backends must keep these signatures."""

    @staticmethod
    def _import_package(package: str, extra: str):
        """Import an optional backend and explain which extra installs it."""
        try:
            return importlib.import_module(package)
        except ImportError as exc:
            raise ImportError(
                f"{package} is required for this logger and is not installed. "
                "dl_toolbox keeps logging backends as separate extras so you "
                "only install the one you use:\n"
                f"  uv add 'dl-toolbox[{extra}]'\n"
                f"  pip install 'dl-toolbox[{extra}]'"
            ) from exc

    @abstractmethod
    def __init__(self, log_dir: str, **kwargs):
        log_dir = process_path(log_dir, dir_ok=True, file_ok=False, must_exist=False)
        self.log_dir = log_dir

    @abstractmethod
    def log_scalar(self, step: int, tag: str, value: float):
        pass

    @abstractmethod
    def log_dict(self, step: int, values: Dict[str, float]):
        pass

    @abstractmethod
    def log_image(self, step: int, tag: str, image: Any, **kwargs):
        pass

    @abstractmethod
    def log_table(self, step: int, tag: str, data: Any, **kwargs):
        pass

    @abstractmethod
    def log_histogram(self, step: int, tag: str, values: Any):
        pass

    @abstractmethod
    def log_config(self, config: Dict, config_file_name: str):
        if not config_file_name.endswith(".json"):
            config_file_name = f"{config_file_name}.json"

        config_path = os.path.join(self.log_dir, config_file_name)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    def save(self, object: Any, file_name: str):
        if isinstance(object, dict):
            file_name = f"{file_name}{('.json' if not file_name.endswith('.json') else '')}"
            file_path = os.path.join(self.log_dir, file_name)
            with open(file_path, "w") as f:
                json.dump(object, f, indent=4)
            return

        file_name = f"{file_name}{('.obj' if not file_name.endswith('.obj') else '')}"
        file_path = os.path.join(self.log_dir, file_name)
        with open(file_path, "wb") as f:
            pickle.dump(object, f)

    @abstractmethod
    def close(self):
        pass
