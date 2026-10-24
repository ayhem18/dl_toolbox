import pytest
import torch

from dl_toolbox.dl_engine.metrics.contracts import (
    KEY_OUTPUT,
    KEY_TARGET,
    MetricInputError,
    slice_metric_inputs,
    validate_tensor_mapping,
)
from dl_toolbox.dl_engine.metrics.classification.classification import (
    binary_classification_accuracy_metric,
)


def test_validate_tensor_mapping_rejects_missing_keys() -> None:
    with pytest.raises(MetricInputError, match="missing"):
        validate_tensor_mapping({KEY_OUTPUT: torch.zeros(2, 1)}, {KEY_OUTPUT, KEY_TARGET})


def test_slice_metric_inputs_for_accuracy_metric() -> None:
    metric = binary_classification_accuracy_metric()
    mapping = {
        KEY_OUTPUT: torch.zeros(4, 1),
        KEY_TARGET: torch.zeros(4, 1),
    }
    sliced = slice_metric_inputs(mapping, metric.required_keys)
    assert set(sliced) == {KEY_OUTPUT, KEY_TARGET}
