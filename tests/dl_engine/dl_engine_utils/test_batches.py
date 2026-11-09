from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dl_toolbox.dl_engine.dl_engine_utils.batches import cycling_batches, mapping_to_device
from dl_toolbox.dl_engine.dl_engine_utils.optimizer_lr import set_optimizer_lr


def test_cycling_batches_restarts_after_exhaustion() -> None:
    data = torch.arange(4).float().unsqueeze(1)
    loader = DataLoader(TensorDataset(data), batch_size=2)
    batches = list(cycling_batches(loader=loader, num_iter=3))
    assert len(batches) == 3
    assert batches[0][0].shape[0] == 2
    assert batches[2][0].shape[0] == 2


def test_cycling_empty_loader_errors() -> None:
    loader = DataLoader(TensorDataset(torch.empty(0, 1)), batch_size=2)
    with pytest.raises(ValueError, match="empty"):
        list(cycling_batches(loader=loader, num_iter=2))


def test_mapping_to_device_moves_tensors_only() -> None:
    mapping = {"x": torch.zeros(2), "n": 3}
    moved = mapping_to_device(mapping=mapping, device=torch.device("cpu"))
    assert moved["x"].device.type == "cpu"
    assert torch.equal(moved["x"], mapping["x"])
    assert moved["n"] == 3


def test_set_optimizer_lr_writes_every_group() -> None:
    net = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(
        [{"params": [net.weight], "lr": 0.1}, {"params": [net.bias], "lr": 0.2}]
    )
    set_optimizer_lr(optimizer=optimizer, lr=1e-3)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-3)
