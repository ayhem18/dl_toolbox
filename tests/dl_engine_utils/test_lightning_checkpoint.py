# PARKED: no contract yet. Not collected by pytest until this package is rewritten.
from __future__ import annotations

import pytest

pytest.skip("Parked: this package has no contract yet.", allow_module_level=True)

from pathlib import Path

import torch
from torch import nn

from dl_toolbox.dl_engine_utils.checkpoint.lightning import (
    NET_PREFIX,
    load_lightning_net_weights,
)


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_load_lightning_net_weights_strips_prefix(tmp_path: Path) -> None:
    src = TinyNet()
    with torch.no_grad():
        src.fc.weight.fill_(0.5)
        src.fc.bias.fill_(-0.25)
    state = src.state_dict()
    prefixed = {}
    keys = list(state)
    n = len(keys)
    for i in range(n):
        key = keys[i]
        prefixed[f"{NET_PREFIX}{key}"] = state[key]
    path = tmp_path / "model.ckpt"
    torch.save({"state_dict": prefixed}, path)
    dest = TinyNet()
    with torch.no_grad():
        dest.fc.weight.zero_()
        dest.fc.bias.zero_()
    load_lightning_net_weights(dest, path, torch.device("cpu"))
    assert torch.equal(dest.fc.weight, src.fc.weight)
    assert torch.equal(dest.fc.bias, src.fc.bias)
