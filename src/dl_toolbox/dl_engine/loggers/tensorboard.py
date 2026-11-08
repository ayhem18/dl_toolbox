"""TensorBoardLogger. Entity is the event tree under log_dir / run_name."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dl_toolbox.base.input_output.json_io import JsonTextForm, dump_object
from dl_toolbox.base.packaging import require
from dl_toolbox.base.parsing.parse_path import parse_absolute_path
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.dl_engine.loggers.logger import (
    LogExistence,
    LogStream,
    Logger,
    LoggerError,
    parse_log_existence,
    parse_log_stream,
    require_logger_open,
    table_as_json,
)


class TensorBoardLogger(Logger):
    def __init__(self, *, log_dir: Path, run_name: str) -> None:
        self._log_dir = parse_absolute_path(str(log_dir), label="log_dir")
        self._run_name = _parse_run_name(run_name)
        self._run_dir = self._log_dir / self._run_name
        self._writer = None
        self._open = False

    def setup(self, *, stream: str, existence: str) -> None:
        if self._open:
            raise LoggerError("logger is already open")
        parsed_stream = parse_log_stream(stream)
        parsed_existence = parse_log_existence(existence)
        if parsed_existence == LogExistence.REQUIRE and not self._run_dir.is_dir():
            raise LoggerError(
                f"TensorBoard run directory does not exist: {self._run_dir}"
            )
        suffix = _filename_suffix(parsed_stream)
        SummaryWriter = _import_summary_writer()
        self._writer = SummaryWriter(
            log_dir=str(self._run_dir),
            filename_suffix=suffix,
        )
        self._open = True

    def log_params(self, params: Mapping[str, str]) -> None:
        writer = self._require_writer()
        writer.add_text(
            "params", dump_object(dict(params), form=JsonTextForm.ASCII_PRETTY)
        )
        writer.flush()

    def log_tags(self, tags: Mapping[str, str]) -> None:
        writer = self._require_writer()
        writer.add_text(
            "tags", dump_object(dict(tags), form=JsonTextForm.ASCII_PRETTY)
        )
        writer.flush()

    def log_scalars(self, values: Mapping[str, float], *, step: int) -> None:
        writer = self._require_writer()
        if not values:
            return
        keys = list(values)
        n = len(keys)
        step_i = int(step)
        for i in range(n):
            key = keys[i]
            if not isinstance(key, str) or not key:
                raise LoggerError(f"metric name must be a non-empty str, got {key!r}")
            writer.add_scalar(key, float(values[key]), global_step=step_i)
        writer.flush()

    def log_image(self, tag: str, image: object, *, step: int) -> None:
        writer = self._require_writer()
        name = parse_non_empty_str(tag, label="tag")
        writer.add_image(name, _to_chw_image(image), global_step=int(step))
        writer.flush()

    def log_table(
        self, tag: str, rows: Sequence[Mapping[str, Any]], *, step: int
    ) -> None:
        self.log_text(tag, table_as_json(rows), step=step)

    def log_text(self, tag: str, text: str, *, step: int) -> None:
        writer = self._require_writer()
        name = parse_non_empty_str(tag, label="tag")
        writer.add_text(name, str(text), global_step=int(step))
        writer.flush()

    def close(self) -> None:
        if not self._open:
            return
        writer = self._writer
        if writer is not None:
            writer.flush()
            writer.close()
        self._writer = None
        self._open = False

    def _require_writer(self):
        require_logger_open(self._open)
        if self._writer is None:
            raise LoggerError("logger.setup() must run before logging")
        return self._writer


def _filename_suffix(stream: str) -> str:
    if stream == LogStream.FIT:
        return ".fit"
    return ".inspect"


def _parse_run_name(run_name: str) -> str:
    name = parse_non_empty_str(run_name, label="run_name")
    if "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"run_name must be one path segment, got {run_name!r}")
    return name


def _import_summary_writer():
    require(package="tensorboard")
    try:
        module = require(package="torch.utils.tensorboard")
        return module.SummaryWriter
    except AttributeError as exc:
        raise ImportError(
            "torch.utils.tensorboard.SummaryWriter could not be imported. "
            "Install the tensorboard extra:\n"
            "  uv add 'dl-toolbox[tensorboard]'\n"
            "  pip install 'dl-toolbox[tensorboard]'"
        ) from exc


def _to_chw_image(image: object):
    import torch

    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
    else:
        tensor = torch.as_tensor(image)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        channels_last = tensor.shape[-1] in (1, 3, 4)
        channels_first = tensor.shape[0] in (1, 3, 4)
        if channels_last and not channels_first:
            tensor = tensor.permute(2, 0, 1)
    else:
        raise LoggerError(
            "image must be (Y_dim, X_dim), (Y_dim, X_dim, C), or "
            f"(C, Y_dim, X_dim), got shape {tuple(tensor.shape)}"
        )
    if tensor.shape[0] == 4:
        tensor = tensor[:3]
    tensor = tensor.float()
    if tensor.numel() > 0 and float(tensor.max()) > 1.0:
        tensor = tensor / 255.0
    return tensor
