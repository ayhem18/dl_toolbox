"""Atomic JSON-object configuration files."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object in {path}, got {type(payload).__name__}"
        )
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path
