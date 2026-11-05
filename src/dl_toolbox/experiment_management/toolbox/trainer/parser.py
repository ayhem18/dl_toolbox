"""Pass-through trainer_config. Produces TrainerConfig."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints, model_validator

from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_numerical import parse_bool, parse_int, parse_positive_int
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import require_exact_keys
from dl_toolbox.dl_engine.training.config import TrainerConfig
from dl_toolbox.experiment_management.toolbox.trainer.visible_gpus import (
    bind_visible_gpus,
)

_NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

_KEYS = (
    "max_epochs",
    "accelerator",
    "devices",
    "precision",
    "enable_progress_bar",
)


def _parse_devices(raw: object) -> int | str:
    if isinstance(raw, bool):
        raise ValueError(f"devices must be an int or str, got {raw!r}")
    if isinstance(raw, int):
        if raw < 1:
            raise ValueError(f"devices must be >= 1 when int, got {raw}")
        return raw
    return parse_non_empty_str(raw, label="devices")


def _parse_precision(raw: object) -> int | str:
    if isinstance(raw, bool):
        raise ValueError(f"precision must be an int or str, got {raw!r}")
    if isinstance(raw, int):
        return parse_int(raw, label="precision")
    return parse_non_empty_str(raw, label="precision")


class TrainerConfigParser:
    def __init__(self, payload: dict) -> None:
        obj = parse_mapping(payload, label="trainer_config")
        require_exact_keys(obj, _KEYS, label="trainer_config")
        accelerator = parse_non_empty_str(obj["accelerator"], label="accelerator")
        devices = _parse_devices(obj["devices"])
        if accelerator == "gpu":
            if not isinstance(devices, int):
                raise ValueError(
                    "trainer_config.devices must be an int (GPU count) when "
                    f"accelerator is 'gpu', got {devices!r}"
                )
            bind_visible_gpus(count=devices)
        self._config = TrainerConfig(
            max_epochs=parse_positive_int(obj["max_epochs"], label="max_epochs"),
            accelerator=accelerator,
            devices=devices,
            precision=_parse_precision(obj["precision"]),
            enable_progress_bar=parse_bool(
                obj["enable_progress_bar"], label="enable_progress_bar"
            ),
        )

    @property
    def trainer_config(self) -> TrainerConfig:
        return self._config


# --- pydantic (optional extra: validation) ---
# JSON shape, exact keys, types, gpu→devices-int. Leftover: bind_visible_gpus
# (process, not parse).


class TrainerConfigDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    max_epochs: Annotated[int, Field(gt=0)]
    accelerator: _NonEmptyStr
    devices: Annotated[int, Field(ge=1)] | _NonEmptyStr
    precision: int | _NonEmptyStr
    enable_progress_bar: StrictBool

    @model_validator(mode="after")
    def gpu_devices_must_be_int(self) -> TrainerConfigDocument:
        if self.accelerator == "gpu" and not isinstance(self.devices, int):
            raise ValueError(
                "trainer_config.devices must be an int (GPU count) when "
                f"accelerator is 'gpu', got {self.devices!r}"
            )
        return self


def trainer_config_from_pydantic(payload: object) -> TrainerConfig:
    document = TrainerConfigDocument.model_validate(payload)
    if document.accelerator == "gpu":
        devices = document.devices
        if not isinstance(devices, int):
            raise ValueError(
                "trainer_config.devices must be an int (GPU count) when "
                f"accelerator is 'gpu', got {devices!r}"
            )
        bind_visible_gpus(count=devices)
    return TrainerConfig(
        max_epochs=document.max_epochs,
        accelerator=document.accelerator,
        devices=document.devices,
        precision=document.precision,
        enable_progress_bar=document.enable_progress_bar,
    )
