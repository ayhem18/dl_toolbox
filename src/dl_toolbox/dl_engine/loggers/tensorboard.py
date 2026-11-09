"""TensorBoardLogger. Entity is the event tree under log_dir / run_name."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dl_toolbox.base.input_output.json_io import JsonTextForm, dump_object
from dl_toolbox.base.packaging import require
from dl_toolbox.base.parsing.parse_path import parse_absolute_path # TODO: verify whether the new file management contract covers this use-case.
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.dl_engine.loggers.logger import (
    LogExistence,
    LogStream,
    Logger,
    LoggerError,
    table_as_json,
    parse_log_stream,
    parse_log_existence,
)

# TYPE_CHECKING is True for the language server and False at runtime.
# Importing this module therefore does not load torch and does not fail
# when the tensorboard extra is missing. Construction still calls
# _import_summary_writer() so a missing extra errors at init, not setup.
if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

# why do it this way ? 
# there are 2 aspects of importing: 
# 1. we do not want to import heavier packages unless needed
# 2. The code is much easier to read and reason about when the IDE language server knows the type of the object. \

# given these 2 constraints: how can we deal with it ?

# one option: suggested by the AI coding agent (Grok 4.6 extra high)
# is to call the _import_summary_writer() the first time it is needed. Very reasonable: perfectly serves the first point
# the 2nd one is trickier. Because that way, the imports do indeed work. However, self._writer is untyped

# one way to make the self._writer typed is 

# """
# try:
#     from torch.utils.tensorboard import SummaryWriter
# catch ImportError:
#     pass
# """

# The question in this case becomes: is the problem the import statement itself
# or the fact that we do not want files unreleated to tensorboard to crash during import.

def _import_summary_writer() -> type[SummaryWriter]:
    require(package="tensorboard")
    module = require(package="torch.utils.tensorboard")
    try:
        return module.SummaryWriter
    except AttributeError as exc:
        raise ImportError(
            "torch.utils.tensorboard.SummaryWriter could not be imported. "
            "Install the tensorboard extra:\n"
            "  uv add 'dl-toolbox[tensorboard]'\n"
            "  pip install 'dl-toolbox[tensorboard]'"
        ) from exc


class TensorBoardLogger(Logger):
    def __init__(self, *, log_dir: Path, run_name: str) -> None:
        # sets up the self._open flag
        super().__init__()

        self._log_dir = parse_absolute_path(str(log_dir), label="log_dir")
        self._run_name = _parse_run_name(run_name)
        self._run_dir = self._log_dir / self._run_name

        self._writer_cls = _import_summary_writer()
        self._writer: SummaryWriter | None = None

    def setup(self, *, stream: str, existence: str) -> None:
        if self._open:
            raise LoggerError("logger is already open")

        parsed_stream = parse_log_stream(stream=stream)
        parsed_existence = parse_log_existence(existence=existence)

        if parsed_existence == LogExistence.REQUIRE and not self._run_dir.is_dir():
            raise LoggerError(
                f"TensorBoard run directory does not exist: {self._run_dir}"
            )

        suffix = self._file_name_suffix(parsed_stream)

        self._writer = self._writer_cls(
            log_dir=str(self._run_dir),
            filename_suffix=suffix,
        )

        self._open = True

    def log_params(self, params: Mapping[str, str]) -> None:
        self._require_logger_open()
        self._writer.add_text(
            "params", dump_object(dict(params), form=JsonTextForm.ASCII_PRETTY)
        )
        self._writer.flush()

    def log_tags(self, tags: Mapping[str, str]) -> None:
        self._require_logger_open()
        self._writer.add_text(
            "tags", dump_object(dict(tags), form=JsonTextForm.ASCII_PRETTY)
        )
        self._writer.flush()

    def log_scalars(self, values: Mapping[str, float], *, step: int) -> None:
        self._require_logger_open()
        if not values:
            return
        keys = list(values)
        n = len(keys)
        step_i = int(step)
        for i in range(n):
            key = keys[i]
            if not isinstance(key, str) or not key:
                raise LoggerError(f"metric name must be a non-empty str, got {key!r}")
            self._writer.add_scalar(key, float(values[key]), global_step=step_i)
        self._writer.flush()

    def log_image(self, tag: str, image: object, *, step: int) -> None:
        self._require_logger_open()
        name = parse_non_empty_str(tag, label="tag")
        self._writer.add_image(name, _to_chw_image(image), global_step=int(step))
        self._writer.flush()

    def log_table(
        self, tag: str, rows: Sequence[Mapping[str, Any]], *, step: int
    ) -> None:
        self.log_text(tag, table_as_json(rows), step=step)

    def log_text(self, tag: str, text: str, *, step: int) -> None:
        self._require_logger_open()
        name = parse_non_empty_str(tag, label="tag")
        self._writer.add_text(name, str(text), global_step=int(step))
        self._writer.flush()

    def close(self) -> None:
        if not self._open:
            return
        self._writer.flush()
        self._writer.close()
        self._writer = None
        self._open = False

    def cleanup(self) -> None:
        if self._open:
            self.close()
        if self._run_dir.is_dir():
            shutil.rmtree(self._run_dir)


    ############################################ private helpers ############################################

    def _file_name_suffix(self, stream: str) -> str:
        if stream == LogStream.FIT:
            return ".fit"
        return ".inspect"



def _parse_run_name(run_name: str) -> str:
    name = parse_non_empty_str(run_name, label="run_name")
    if "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"run_name must be one path segment, got {run_name!r}")
    return name


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
