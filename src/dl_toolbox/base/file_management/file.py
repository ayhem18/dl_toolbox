"""``File`` is an existing absolute regular file.

Existence is an invariant. Write is not parse: bytes land under a Directory
and the result is a File.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dl_toolbox.base.file_management.join import join_under
from dl_toolbox.base.file_management.parent_walk import find_file_in_parents
from dl_toolbox.base.file_management.paths import abs_path

if TYPE_CHECKING:
    from dl_toolbox.base.file_management.directory import Directory


class File:
    """Existing absolute regular file. Construction validates."""

    def __init__(self, *, path: Path) -> None:
        
        if not isinstance(path, Path):
            raise TypeError(f"path must be pathlib.Path, got {type(path)!r}")
        if not path.is_absolute():
            raise ValueError(f"path must be absolute (caller-owned), got {path}")
        if path.is_file():
            self._path = path
            return
        if path.exists():
            raise ValueError(f"path must be a file, got directory {path}")
        raise FileNotFoundError(f"file must exist: {path}")

    @property
    def path(self) -> Path:
        return self._path

    @classmethod
    def from_absolute(cls, *, path: Path | str) -> File:
        """Absolute-path source. The path must already be a file."""
        return cls(path=abs_path(path))

    @classmethod
    def from_under(cls, *, root: Directory, relative_path: str) -> File:
        """Root + posix-relative source. ``root`` must exist. No ``..``."""
        from dl_toolbox.base.file_management.directory import Directory, DirectorySpec

        if not isinstance(root, Directory):
            raise TypeError(f"root must be Directory, got {type(root)!r}")
        if root.spec != DirectorySpec.EXISTS:
            raise ValueError(f"root Directory must exist: {root.path}")
        return cls.from_absolute(path=join_under(root.path, relative_path))

    @classmethod
    def from_parent_walk(cls, *, start: File | Directory, basename: str) -> File:
        """Walk ancestors from ``start`` until ``basename`` is a file."""
        from dl_toolbox.base.file_management.directory import Directory, DirectorySpec

        if isinstance(start, File):
            walk_from = start.parent().path
        elif isinstance(start, Directory):
            if start.spec != DirectorySpec.EXISTS:
                raise ValueError(f"walk start Directory must exist: {start.path}")
            walk_from = start.path
        else:
            raise TypeError(f"start must be File or Directory, got {type(start)!r}")
        found = find_file_in_parents(basename, start=walk_from)
        return cls.from_absolute(path=found)

    def parent(self) -> Directory:
        """Named convert: this file's parent directory (exists)."""
        from dl_toolbox.base.file_management.directory import (
            Directory,
            DirectoryParsePolicy,
        )

        return Directory.from_absolute(
            path=self._path.parent,
            policy=DirectoryParsePolicy.MUST_EXIST,
        )

    def __fspath__(self) -> str:
        return str(self._path)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, File) and self._path == other._path

    def __hash__(self) -> int:
        return hash(self._path)

    def __repr__(self) -> str:
        return f"File(path={self._path!r})"
