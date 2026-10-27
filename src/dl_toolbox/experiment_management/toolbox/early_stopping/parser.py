"""Pass-through early_stopping_config. Off is name=none, not a missing key."""

from __future__ import annotations

from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_numerical import parse_non_negative_number, parse_positive_int
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import require_exact_keys
from dl_toolbox.dl_engine.training.config import EarlyStoppingConfig

_KEYS = ("name", "params")
_PATIENCE_KEYS = ("patience", "min_delta")


class EarlyStoppingConfigParser:
    def __init__(self, payload: dict) -> None:
        obj = parse_mapping(payload, label="early_stopping_config")
        require_exact_keys(obj, _KEYS, label="early_stopping_config")
        name = parse_non_empty_str(obj["name"], label="name")
        params = parse_mapping(obj["params"], label="params")
        if name == EarlyStoppingConfig.NONE:
            require_exact_keys(params, (), label="params")
            self._config = EarlyStoppingConfig.none()
            return
        if name == EarlyStoppingConfig.PATIENCE:
            require_exact_keys(params, _PATIENCE_KEYS, label="params")
            self._config = EarlyStoppingConfig(
                name=EarlyStoppingConfig.PATIENCE,
                patience=parse_positive_int(params["patience"], label="patience"),
                min_delta=parse_non_negative_number(
                    params["min_delta"], label="min_delta"
                ),
            )
            return
        raise ValueError(
            "early_stopping_config.name must be "
            f"{EarlyStoppingConfig.NONE!r} or {EarlyStoppingConfig.PATIENCE!r}, "
            f"got {name!r}"
        )

    @property
    def early_stopping_config(self) -> EarlyStoppingConfig:
        return self._config
