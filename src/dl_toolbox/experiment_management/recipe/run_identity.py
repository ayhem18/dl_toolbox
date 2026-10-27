"""Claimed run: payload folder, label, configuration hash."""

from __future__ import annotations

from pathlib import Path

from dl_toolbox.base.parsing.parse_str import parse_non_empty_str


class RunIdentity:
    def __init__(
        self,
        *,
        run_dir: Path,
        human_readable_name: str,
        configuration_hash: str,
    ) -> None:
        if not run_dir.is_absolute():
            raise ValueError(f"run_dir must be an absolute path, got {run_dir}")
        self.run_dir = run_dir
        self.human_readable_name = parse_non_empty_str(
            human_readable_name, label="human_readable_name"
        )
        self.hash = parse_non_empty_str(configuration_hash, label="configuration_hash")
