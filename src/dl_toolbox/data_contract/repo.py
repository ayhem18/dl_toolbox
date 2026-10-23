"""Walk parents from an absolute start. Never use ``Path.cwd()``."""

from __future__ import annotations

import os
from pathlib import Path


def _start_directory(start: Path | str) -> Path:
    path = Path(start)
    if not path.is_absolute():
        raise ValueError(f"start must be an absolute path, got {path!s}")
    path = path.resolve()
    if path.is_file():
        return path.parent
    return path


def _assert_basename(name: str, *, kind: str) -> str:
    text = str(name).strip()
    if not text:
        raise ValueError(f"{kind} must be a non-empty str")
    if "/" in text.replace("\\", "/") or text in {".", ".."}:
        raise ValueError(f"{kind} must be a basename: {text!r}")
    return text


def find_named_directory(name: str, *, start: Path | str) -> Path:
    """Walk up from ``start`` until a directory named ``name`` appears."""
    dirname = _assert_basename(name, kind="directory name")
    current_dir = _start_directory(start)
    while True:
        try:
            names = os.listdir(current_dir)
        except OSError as exc:
            raise FileNotFoundError(
                f"Cannot list directory while searching for {dirname!r}: {current_dir}"
            ) from exc
        if dirname in names:
            found = current_dir / dirname
            if not found.is_dir():
                raise FileNotFoundError(
                    f"Found {dirname!r} but it is not a directory: {found}"
                )
            return found
        parent_dir = current_dir.parent
        if parent_dir == current_dir:
            raise FileNotFoundError(
                f"Could not find directory {dirname!r} in any parent of {current_dir}"
            )
        current_dir = parent_dir


def find_file_in_parents(filename: str, *, start: Path | str) -> Path:
    """Walk up from ``start`` until a file named ``filename`` appears."""
    name = _assert_basename(filename, kind="file name")
    current_dir = _start_directory(start)
    while True:
        try:
            names = os.listdir(current_dir)
        except OSError as exc:
            raise FileNotFoundError(
                f"Cannot list directory while searching for {name!r}: {current_dir}"
            ) from exc
        if name in names:
            found = current_dir / name
            if not found.is_file():
                raise FileNotFoundError(
                    f"Found {name!r} but it is not a file: {found}"
                )
            return found
        parent_dir = current_dir.parent
        if parent_dir == current_dir:
            raise FileNotFoundError(
                f"Could not find file {name!r} in any parent of {current_dir}"
            )
        current_dir = parent_dir
