"""TensorBoard training logger. Holds the SummaryWriter."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.dl_engine.loggers.base_logger import (
    STREAM_FIT,
    STREAM_INSPECT,
    Logger,
    LoggerError,
    parse_log_stream,
    require_logger_open,
    table_as_json,
)
from dl_toolbox.dl_engine.loggers.tensorboard.tensorboard_manager import TensorBoardManager


class TensorBoardLogger(Logger):
    def __init__(self, *, manager: TensorBoardManager, run_name: str) -> None:
        self._manager = manager
        self.run_name = parse_non_empty_str(run_name, label="run_name")
        self._writer = None
        self._open = False

    def setup(self, *, stream: str = STREAM_FIT) -> None:
        if self._open:
            raise LoggerError("logger is already open")
        parsed = parse_log_stream(stream)
        suffix = ".inspect" if parsed == STREAM_INSPECT else ""
        self._writer = self._manager.create_writer(
            self.run_name, filename_suffix=suffix
        )
        self._open = True

    def log_params(self, params: Mapping[str, str]) -> None:
        writer = self._require_writer()
        writer.add_text("params", json.dumps(dict(params), indent=2, ensure_ascii=True))
        writer.flush()

    def log_tags(self, tags: Mapping[str, str]) -> None:
        writer = self._require_writer()
        writer.add_text("tags", json.dumps(dict(tags), indent=2, ensure_ascii=True))
        writer.flush()

    def log_scalars(self, values: Mapping[str, float], *, step: int) -> None:
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
        if self._writer is not None:
            self._writer.close()
        self._writer = None
        self._open = False

    def _require_writer(self):
        require_logger_open(self._open)
        if self._writer is None:
            raise LoggerError("logger.setup() must run before logging")
        return self._writer


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
            f"image must be HW, HWC, or CHW, got shape {tuple(tensor.shape)}"
        )
    if tensor.shape[0] == 4:
        tensor = tensor[:3]
    tensor = tensor.float()
    if tensor.numel() > 0 and float(tensor.max()) > 1.0:
        tensor = tensor / 255.0
    return tensor
