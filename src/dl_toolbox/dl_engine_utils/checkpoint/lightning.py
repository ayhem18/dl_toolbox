"""Load Lightning ClassificationLightningModule checkpoints into a raw nn.Module."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

NET_PREFIX = "net."


def load_lightning_net_weights(
    model: nn.Module, ckpt_path: Path, device: torch.device
) -> None:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    if not isinstance(state, dict):
        raise TypeError(
            f"checkpoint state_dict must be a dict, got {type(state)!r}"
        )
    keys = list(state)
    n_keys = len(keys)
    stripped: dict[str, Tensor] = {}
    for i in range(n_keys):
        key = keys[i]
        if not isinstance(key, str):
            raise TypeError(f"checkpoint key must be str, got {type(key)!r}")
        if key.startswith(NET_PREFIX):
            stripped[key[len(NET_PREFIX) :]] = state[key]
    if len(stripped) == 0:
        raise ValueError(
            f"checkpoint has no {NET_PREFIX!r} weights: {ckpt_path}"
        )
    missing, unexpected = model.load_state_dict(stripped, strict=True)
    if missing:
        raise ValueError(f"missing checkpoint keys: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"unexpected checkpoint keys: {sorted(unexpected)}")
