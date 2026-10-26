"""dl-toolbox command line. cwd is allowed only here."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dl_toolbox.project_layout.layout import DEFAULT_TRACKING_BACKEND, TRACKING_BACKENDS
from dl_toolbox.project_layout.scaffold import setup_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dl-toolbox")
    subparsers = parser.add_subparsers(dest="command", required=True)
    setup = subparsers.add_parser(
        "setup-project",
        help="Create an opinionated toolbox project under the current directory.",
    )
    setup.add_argument("project_name")
    setup.add_argument(
        "--package",
        default=None,
        help="Optional Python package name. Writes a nested pyproject.toml, not one at the project root.",
    )
    setup.add_argument(
        "--tracking-backend",
        default=DEFAULT_TRACKING_BACKEND,
        choices=sorted(TRACKING_BACKENDS),
        help="Selected tracking backend. Only this backend gets a storage folder; configs for every backend are written.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command != "setup-project":
        parser.error(f"unknown command {args.command!r}")
    parent = Path.cwd().resolve()
    try:
        root = setup_project(
            args.project_name,
            parent=parent,
            package=args.package,
            tracking_backend=args.tracking_backend,
        )
    except (ValueError, FileExistsError, FileNotFoundError, TypeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(f"{root}\n")
    return 0
