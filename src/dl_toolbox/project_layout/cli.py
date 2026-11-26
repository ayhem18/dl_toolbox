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
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.project_layout.names import MARKER_FILENAME, TRACKING_BACKEND_FILENAME
from dl_toolbox.project_layout.project_layout import ProjectLayout, TrackingBackend
from dl_toolbox.remote_store.backends import RemoteBackend


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
    setup.add_argument(
        "--remote-backend",
        default=RemoteBackend.S3,
        choices=sorted(RemoteBackend.BACKENDS),
        help=(
            "Backend recorded in remote/remote.json (default: s3). "
            "A blank credential file is scaffolded for every backend under "
            "remote/credentials/, which .gitignore excludes."
        ),
    )

    push = subparsers.add_parser(
        "remote-push",
        help="Push a file or directory under storage/ to the remote store.",
    )
    push.add_argument(
        "path", help="A file or directory under the project's storage/."
    )

    pull = subparsers.add_parser(
        "remote-pull",
        help="Pull a file or directory under storage/ from the remote (overwrites).",
    )
    pull.add_argument("path", help="A path under the project's storage/.")
    pull.add_argument(
        "--dir",
        dest="is_dir",
        action="store_true",
        help="Treat path as a directory instead of a file.",
    )

    exists = subparsers.add_parser(
        "remote-exists",
        help="Check whether a file is present in the remote store (prints present/absent).",
    )
    exists.add_argument("path", help="A file under the project's storage/.")

    return parser


def _project_folder_name(raw: object) -> str:
    name = parse_non_empty_str(raw, label="project_name")
    normalized = name.replace("\\", "/")
    if "/" in normalized or normalized in {".", ".."}:
        raise ValueError(f"project_name must be a basename, got {name!r}")
    return name


def _setup_from_cwd(
    *, project_name: str, tracking_backend: str, remote_backend: str
) -> Path:
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
            remote_backend=remote_backend,
        )
    except Exception:
        shutil.rmtree(root.path)
        raise
    return root.path


def _cwd_directory() -> Directory:
    return Directory.from_absolute(
        path=Path.cwd().resolve(), policy=DirectoryParsePolicy.MUST_EXIST
    )


def _layout_from_cwd() -> ProjectLayout:
    """Locate the project from cwd and build its layout.

    The tracking backend is read from the scaffolded ``.tracking-backend`` so a
    remote command does not have to name a backend it does not use.
    """
    marker = File.from_parent_walk(start=_cwd_directory(), basename=MARKER_FILENAME)
    backend_path = marker.parent().path / TRACKING_BACKEND_FILENAME
    if not backend_path.is_file():
        raise FileNotFoundError(
            f"missing {TRACKING_BACKEND_FILENAME} at project root "
            f"{marker.parent().path}"
        )
    tracking_backend = backend_path.read_text(encoding="utf-8").strip()
    return ProjectLayout.from_marker(start=marker, tracking_backend=tracking_backend)


def _absolute(path_arg: str) -> Path:
    candidate = Path(path_arg)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def _relative_under_storage(layout: ProjectLayout, path_arg: str) -> str:
    storage_root = layout.storage().path.resolve()
    absolute = _absolute(path_arg)
    try:
        relative = absolute.relative_to(storage_root)
    except ValueError as exc:
        raise ValueError(
            f"{path_arg} is not under storage/ ({storage_root})"
        ) from exc
    return relative.as_posix()


def _remote_push(*, path: str) -> int:
    absolute = _absolute(path)
    is_directory = absolute.is_dir()
    layout = _layout_from_cwd()
    mirror = layout.remote_mirror()
    if is_directory:
        relative = _relative_under_storage(layout, path)
        mirror.push_directory(relative_path=relative)
        sys.stdout.write(f"{relative}\n")
        return 0
    relative = mirror.push_file(local_file=File.from_absolute(path=absolute))
    sys.stdout.write(f"{relative}\n")
    return 0


def _remote_pull(*, path: str, is_dir: bool) -> int:
    layout = _layout_from_cwd()
    mirror = layout.remote_mirror()
    relative = _relative_under_storage(layout, path)
    if is_dir:
        mirror.pull_directory(relative_path=relative)
        sys.stdout.write(f"{relative}\n")
        return 0
    pulled = mirror.pull_file(relative_path=relative)
    sys.stdout.write(f"{pulled.path}\n")
    return 0


def _remote_exists(*, path: str) -> int:
    layout = _layout_from_cwd()
    mirror = layout.remote_mirror()
    relative = _relative_under_storage(layout, path)
    present = mirror.exists(relative_path=relative)
    sys.stdout.write("present\n" if present else "absent\n")
    return 0


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "setup-project":
        root = _setup_from_cwd(
            project_name=args.project_name,
            tracking_backend=args.tracking_backend,
            remote_backend=args.remote_backend,
        )
        sys.stdout.write(f"{root}\n")
        return 0
    if args.command == "remote-push":
        return _remote_push(path=args.path)
    if args.command == "remote-pull":
        return _remote_pull(path=args.path, is_dir=args.is_dir)
    if args.command == "remote-exists":
        return _remote_exists(path=args.path)
    raise ValueError(f"unknown command {args.command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return _dispatch(args)
    except Exception as exc:  # CLI boundary: surface the message, not a traceback
        sys.stderr.write(f"{exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
