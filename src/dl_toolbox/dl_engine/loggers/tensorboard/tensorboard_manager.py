"""TensorBoard store: log_dir and SummaryWriter construction. No log_*."""

from __future__ import annotations

import importlib
from pathlib import Path

from dl_toolbox.base.parsing.parse_path import parse_absolute_path
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.dl_engine.loggers.base_logger import import_extra
from dl_toolbox.dl_engine.loggers.base_manager import LoggingManager

_EXTRA = "tensorboard"


class TensorBoardManager(LoggingManager):
    def __init__(self, *, log_dir: Path) -> None:
        self._log_dir = parse_absolute_path(str(log_dir), label="log_dir")

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    def prepare(self) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_name: str) -> Path:
        name = parse_non_empty_str(run_name, label="run_name")
        return self._log_dir / name

    def create_writer(self, run_name: str, *, filename_suffix: str = ""):
        self.prepare()
        run_dir = self.run_dir(run_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        SummaryWriter = self._import_summary_writer()
        return SummaryWriter(log_dir=str(run_dir), filename_suffix=filename_suffix)

    @staticmethod
    def _import_summary_writer():
        import_extra("tensorboard", extra=_EXTRA)
        import_extra("torch", extra=_EXTRA)
        try:
            module = importlib.import_module("torch.utils.tensorboard")
            return module.SummaryWriter
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "torch.utils.tensorboard.SummaryWriter could not be imported. "
                "Install the tensorboard extra:\n"
                f"  uv add 'dl-toolbox[{_EXTRA}]'\n"
                f"  pip install 'dl-toolbox[{_EXTRA}]'"
            ) from exc
