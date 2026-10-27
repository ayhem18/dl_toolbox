"""Pass-through trainer_config. Produces TrainerConfig."""

from __future__ import annotations

from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_numerical import parse_bool, parse_int, parse_positive_int
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import require_exact_keys
from dl_toolbox.dl_engine.training.config import TrainerConfig

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
        self._config = TrainerConfig(
            max_epochs=parse_positive_int(obj["max_epochs"], label="max_epochs"),
            accelerator=parse_non_empty_str(obj["accelerator"], label="accelerator"),
            devices=_parse_devices(obj["devices"]),
            precision=_parse_precision(obj["precision"]),
            enable_progress_bar=parse_bool(
                obj["enable_progress_bar"], label="enable_progress_bar"
            ),
        )

    @property
    def trainer_config(self) -> TrainerConfig:
        return self._config
