"""dl-toolbox command line. cwd is allowed only here."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
    DirectorySpec,
)
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.project_layout.project_layout import ProjectLayout, TrackingBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dl-toolbox")
    subparsers = parser.add_subparsers(dest="command", required=True)
    setup = subparsers.add_parser(
        "setup-project",
        help="Create a toolbox project directory under cwd and write the tree.",
    )
    setup.add_argument("project_name")
    setup.add_argument(
        "--tracking-backend",
        required=True,
        choices=sorted(TrackingBackend.BACKENDS),
        help=(
            "Scaffolded default written to .tracking-backend. "
            "from_start still takes tracking_backend per process. "
            "Configs for tensorboard and mlflow are both written."
        ),
    )
    return parser


def _project_folder_name(raw: object) -> str:
    name = parse_non_empty_str(raw, label="project_name")
    normalized = name.replace("\\", "/")
    if "/" in normalized or normalized in {".", ".."}:
        raise ValueError(f"project_name must be a basename, got {name!r}")
    return name


def _setup_from_cwd(*, project_name: str, tracking_backend: str) -> Path:
    name = _project_folder_name(project_name)
    parent = Directory.from_absolute(
        path=Path.cwd().resolve(),
        policy=DirectoryParsePolicy.MUST_EXIST,
    )
    proposed = Directory.from_under(
        root=parent,
        relative_path=name,
        policy=DirectoryParsePolicy.ALLOW_MISSING,
    )
    if proposed.spec == DirectorySpec.EXISTS:
        raise FileExistsError(f"project directory already exists: {proposed.path}")
    root = Directory.from_absolute(
        path=proposed.path, policy=DirectoryParsePolicy.CREATE
    )
    try:
        ProjectLayout.scaffold(
            root=root,
            project_name=name,
            tracking_backend=tracking_backend,
        )
    except Exception:
        shutil.rmtree(root.path)
        raise
    return root.path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command != "setup-project":
        parser.error(f"unknown command {args.command!r}")
    try:
        root = _setup_from_cwd(
            project_name=args.project_name,
            tracking_backend=args.tracking_backend,
        )
    except (ValueError, FileExistsError, FileNotFoundError, TypeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(f"{root}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
