"""Injected locations. Core does not walk the project tree."""

from __future__ import annotations

from pathlib import Path

from dl_toolbox.base.parsing.parse_path import parse_absolute_path


class EnginePaths:
    def __init__(
        self,
        *,
        project_root: Path,
        records_dir: Path,
        payload_root: Path,
    ) -> None:
        self.project_root = parse_absolute_path(str(project_root), label="project_root")
        self.records_dir = parse_absolute_path(str(records_dir), label="records_dir")
        self.payload_root = parse_absolute_path(str(payload_root), label="payload_root")

    def payload_dir(self, name: str) -> Path:
        return self.payload_root / name
