"""TensorBoard event-file persistence. Import tensorboard only here."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path

from dl_toolbox.dl_engine.loggers.base_logger import (
    Logger,
    LoggerError,
    copy_local_artifact,
    import_extra,
    require_logger_open,
)
from dl_toolbox.dl_engine.loggers.base_manager import (
    LoggingManager,
    logger_run_name,
    publish_run_tree,
)
from dl_toolbox.base.parsing.parse_path import parse_absolute_path
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str

_EXTRA = "tensorboard"


class TensorBoardManager(LoggingManager):
    def __init__(self, *, log_dir: Path) -> None:
        self._log_dir = parse_absolute_path(str(log_dir), label="log_dir")
        self._run_dir: Path | None = None
        self._writer = None
        self._open = False

    @property
    def log_dir(self) -> Path:
        return self._log_dir

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

    def setup(self, *, run_name: str) -> None:
        if self._open:
            raise LoggerError("logger is already open")
        name = parse_non_empty_str(run_name, label="run_name")
        run_dir = self._log_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        SummaryWriter = self._import_summary_writer()
        self._run_dir = run_dir
        self._writer = SummaryWriter(log_dir=str(run_dir))
        self._open = True

    def log_params(self, params: Mapping[str, str]) -> None:
        writer = self._require_writer()
        writer.add_text("params", json.dumps(dict(params), indent=2, ensure_ascii=True))
        writer.flush()

    def log_tags(self, tags: Mapping[str, str]) -> None:
        writer = self._require_writer()
        writer.add_text("tags", json.dumps(dict(tags), indent=2, ensure_ascii=True))
        writer.flush()

    def log_metrics(self, values: Mapping[str, float], *, step: int) -> None:
        writer = self._require_writer()
        if not values:
            return
        keys = list(values)
        n = len(keys)
        for i in range(n):
            key = keys[i]
            if not isinstance(key, str) or not key:
                raise LoggerError(f"metric name must be a non-empty str, got {key!r}")
            writer.add_scalar(key, float(values[key]), global_step=int(step))
        writer.flush()

    def log_artifact(self, path: Path | str, *, dest: str | None = None) -> None:
        self._require_writer()
        if self._run_dir is None:
            raise LoggerError("logger.setup() must run before logging")
        copy_local_artifact(path, self._run_dir, dest=dest)

    def close(self) -> None:
        if not self._open:
            return
        if self._writer is not None:
            self._writer.close()
        self._writer = None
        self._open = False

    def publish_directory(
        self,
        local_dir: Path,
        *,
        dest: str,
        logger: Logger,
    ) -> None:
        publish_run_tree(
            local_dir,
            dest=dest,
            parent=self._log_dir,
            run_name=logger_run_name(logger),
        )

    def _require_writer(self):
        require_logger_open(self._open)
        if self._writer is None:
            raise LoggerError("logger.setup() must run before logging")
        return self._writer
