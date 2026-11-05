"""Weights-only checkpoint payload. Pass ``weights_writer`` as ``write_function``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch


def write_weights(*, path: Path, state_dict: Mapping[str, Any]) -> None:
    """Write ``{"state_dict": ...}`` with ``torch.save``.

    Keys are stored as given. A Lightning module's ``state_dict`` keeps the
    ``net.`` prefix that ``load_lightning_net_weights`` expects.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": dict(state_dict)}, path)


def weights_writer(
    *, state_dict: Mapping[str, Any]
) -> Callable[[Path], None]:
    snapshot = dict(state_dict)

    def write(path: Path) -> None:
        write_weights(path=path, state_dict=snapshot)

    return write
