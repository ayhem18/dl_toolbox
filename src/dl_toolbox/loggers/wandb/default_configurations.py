"""Opinionated W&B setup dicts. Paths stay caller-owned."""

from __future__ import annotations

from pathlib import Path


def default_wandb_setup(*, project: str, dir: str | Path) -> dict:
    return {"project": project, "dir": str(dir)}
