"""Join a relative payload path under a collection artifact directory."""

from __future__ import annotations

from pathlib import Path


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
