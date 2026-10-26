"""Opinionated W&B store dicts. Paths stay caller-owned. Project is identity."""

from __future__ import annotations

from pathlib import Path


def default_wandb_setup(*, dir: str | Path) -> dict:
    return {"dir": str(dir)}
