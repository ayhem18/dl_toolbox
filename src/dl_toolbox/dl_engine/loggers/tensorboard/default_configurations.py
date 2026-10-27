"""Opinionated TensorBoard setup dicts. Paths stay caller-owned."""

from __future__ import annotations

from pathlib import Path


def default_tensorboard_setup(*, log_dir: str | Path) -> dict:
    return {"log_dir": str(log_dir)}
