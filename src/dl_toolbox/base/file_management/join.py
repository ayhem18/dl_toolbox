"""Join a posix-relative path under an absolute root. Never use cwd."""

from __future__ import annotations

from pathlib import Path

from dl_toolbox.base.parsing.parse_str import parse_non_empty_str


def join_under(directory: Path | str, relative_path: str) -> Path:
    """Return an existing absolute file that stays under ``directory``.

    ``directory`` must already be absolute. This function does not use cwd.
    """
    root = Path(directory)
    if not root.is_absolute():
        raise ValueError(f"directory must be an absolute path, got {root!s}")
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"expected an existing absolute directory: {root}")
    relative = str(relative_path).replace("\\", "/").strip()
    if not relative:
        raise ValueError("relative_path is empty")
    if relative.startswith("/") or Path(relative).is_absolute():
        raise ValueError(f"relative_path must be relative: {relative_path!r}")
    parts = [part for part in relative.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"relative_path must stay under the directory: {relative_path!r}")
    absolute_path = (root.joinpath(*parts)).resolve()
    if not absolute_path.is_absolute() or not absolute_path.is_file():
        raise FileNotFoundError(f"expected an existing absolute file: {absolute_path}")
    try:
        absolute_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"path escapes artifact directory: {relative_path!r}"
        ) from exc
    return absolute_path


def resolve_under_project(project_root: Path, relative_path: str, *, label: str) -> Path:
    """Join a posix-relative path under an absolute project root.

    The target does not have to exist. ``..`` and absolute paths are rejected.
    This function does not use cwd.
    """
    if not project_root.is_absolute():
        raise ValueError(f"project_root must be an absolute path, got {project_root}")
    root = project_root.resolve()
    text = parse_non_empty_str(relative_path, label=label).replace("\\", "/")
    if Path(text).is_absolute() or text.startswith("/"):
        raise ValueError(f"{label} must be relative to the project root, got {relative_path!r}")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts:
        raise ValueError(f"{label} is empty after normalization: {relative_path!r}")
    n_parts = len(parts)
    for i in range(n_parts):
        if parts[i] == "..":
            raise ValueError(
                f"{label} must stay under the project root, got {relative_path!r}"
            )
    absolute = root.joinpath(*parts).resolve()
    try:
        absolute.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{label} escapes the project root: {relative_path!r}"
        ) from exc
    return absolute
