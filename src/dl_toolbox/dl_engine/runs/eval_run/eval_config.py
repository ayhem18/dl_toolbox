"""EvalRun grouped inputs. LogItem constructors reject illegal payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, field_validator
from torch import nn
from torch.utils.data import DataLoader

from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.dl_engine.loggers.logger import parse_log_existence, parse_log_stream

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)


def _require_step(step: object) -> int:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be an int >= 0, got {step!r}")
    return step


class EvalLoggerConfig(BaseModel):
    model_config = _STRICT

    stream: Literal["fit", "inspect"]
    existence: Literal["create", "require"]
    step: int

    @field_validator("stream")
    @classmethod
    def _stream(cls, value: object) -> str:
        return parse_log_stream(value)

    @field_validator("existence")
    @classmethod
    def _existence(cls, value: object) -> str:
        return parse_log_existence(value)

    @field_validator("step")
    @classmethod
    def _step(cls, value: object) -> int:
        return _require_step(value)


class EvalLoaders:
    def __init__(self, *, named: Mapping[str, DataLoader]) -> None:
        if not isinstance(named, Mapping):
            raise TypeError(f"named must be a mapping, got {type(named)!r}")
        loaders: dict[str, DataLoader] = {}
        names = list(named)
        n_names = len(names)
        if n_names < 1:
            raise ValueError("EvalLoaders must have at least one DataLoader")
        for i in range(n_names):
            name = parse_non_empty_str(names[i], label="loader name")
            loader = named[names[i]]
            if not isinstance(loader, DataLoader):
                raise TypeError(
                    f"loaders[{name!r}] must be a DataLoader, got {type(loader)!r}"
                )
            if name in loaders:
                raise ValueError(f"EvalLoaders has duplicate name {name!r}")
            loaders[name] = loader
        self.named = loaders


class EvalContext:
    def __init__(
        self,
        *,
        net: nn.Module,
        loaders: EvalLoaders,
        device: torch.device,
        step: int,
    ) -> None:
        if not isinstance(net, nn.Module):
            raise TypeError(f"net must be nn.Module, got {type(net)!r}")
        if not isinstance(loaders, EvalLoaders):
            raise TypeError(f"loaders must be EvalLoaders, got {type(loaders)!r}")
        if not isinstance(device, torch.device):
            raise TypeError(f"device must be torch.device, got {type(device)!r}")
        self.net = net
        self.loaders = loaders
        self.device = device
        self.step = _require_step(step)


class LogItem:
    SCALARS = "scalars"
    IMAGE = "image"
    TEXT = "text"
    TABLE = "table"

    def __init__(self) -> None:
        raise TypeError("use LogItem.scalars / image / text / table")

    @classmethod
    def scalars(cls, *, step: int, values: Mapping[str, float]) -> Self:
        if not isinstance(values, Mapping):
            raise TypeError(f"values must be a mapping, got {type(values)!r}")
        parsed: dict[str, float] = {}
        keys = list(values)
        n_keys = len(keys)
        for i in range(n_keys):
            key = parse_non_empty_str(keys[i], label="scalar name")
            value = values[keys[i]]
            if type(value) is not float:
                raise TypeError(
                    f"values[{key!r}] must be a float, got {type(value)!r}"
                )
            parsed[key] = value
        item = cls.__new__(cls)
        item.kind = cls.SCALARS
        item.step = _require_step(step)
        item.values = parsed
        return item

    @classmethod
    def image(cls, *, step: int, tag: str, image: object) -> Self:
        item = cls.__new__(cls)
        item.kind = cls.IMAGE
        item.step = _require_step(step)
        item.tag = parse_non_empty_str(tag, label="tag")
        item.image = image
        return item

    @classmethod
    def text(cls, *, step: int, tag: str, text: str) -> Self:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text)!r}")
        item = cls.__new__(cls)
        item.kind = cls.TEXT
        item.step = _require_step(step)
        item.tag = parse_non_empty_str(tag, label="tag")
        item.text = text
        return item

    @classmethod
    def table(
        cls, *, step: int, tag: str, rows: Sequence[Mapping[str, Any]]
    ) -> Self:
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise TypeError(f"rows must be a sequence of mappings, got {type(rows)!r}")
        parsed: list[dict[str, Any]] = []
        n_rows = len(rows)
        for i in range(n_rows):
            row = rows[i]
            if not isinstance(row, Mapping):
                raise TypeError(f"rows[{i}] must be a mapping, got {type(row)!r}")
            parsed.append(dict(row))
        item = cls.__new__(cls)
        item.kind = cls.TABLE
        item.step = _require_step(step)
        item.tag = parse_non_empty_str(tag, label="tag")
        item.rows = parsed
        return item
