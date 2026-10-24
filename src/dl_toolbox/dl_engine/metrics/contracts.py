from __future__ import annotations

from collections.abc import Iterable, Mapping

from torch import Tensor

# Standard tensor keys for classification metrics in dl_toolbox.
KEY_OUTPUT = "output"
KEY_TARGET = "target"

ACCURACY_KEYS = frozenset({KEY_OUTPUT, KEY_TARGET})
LOSS_KEYS = frozenset({KEY_OUTPUT, KEY_TARGET})


class MetricInputError(ValueError):
    pass


def validate_mapping_has_keys(mapping: Mapping, required_keys: Iterable[str]) -> None:
    missing = set(required_keys) - set(mapping.keys())
    if missing:
        raise MetricInputError(f"missing required keys {sorted(missing)}; got {sorted(mapping)}")


def validate_tensor_mapping(mapping: Mapping, required_keys: Iterable[str]) -> None:
    validate_mapping_has_keys(mapping, required_keys)
    for key in required_keys:
        if not isinstance(mapping[key], Tensor):
            raise MetricInputError(
                f"key {key!r} must be a torch.Tensor, got {type(mapping[key]).__name__}"
            )


def slice_metric_inputs(
    mapping: Mapping[str, Tensor],
    required_keys: Iterable[str],
) -> dict[str, Tensor]:
    validate_tensor_mapping(mapping, required_keys)
    return {key: mapping[key] for key in required_keys}
