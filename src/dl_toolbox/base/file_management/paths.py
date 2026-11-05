"""Absolute paths with a declared role and presence.

Callers own resolution. This module does not use ``Path.cwd()``.
"""

from __future__ import annotations

from pathlib import Path


class PathRole:
    """Declared kind. Not inferred from the filesystem after the fact."""

    def __init__(self, *, name: str) -> None:
        if name not in ("file", "directory"):
            raise ValueError(f"path role must be 'file' or 'directory', got {name!r}")
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PathRole) and self._name == other._name

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"PathRole.{self._name.upper()}"


PathRole.FILE = PathRole(name="file")
PathRole.DIRECTORY = PathRole(name="directory")


class PathPresence:
    """What to do if the path is missing, and what is allowed if it exists."""

    def __init__(self, *, name: str) -> None:
        if name not in ("must_exist", "create", "allow_missing"):
            raise ValueError(
                "path presence must be 'must_exist', 'create', or "
                f"'allow_missing', got {name!r}"
            )
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PathPresence) and self._name == other._name

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"PathPresence.{self._name.upper()}"


PathPresence.MUST_EXIST = PathPresence(name="must_exist")
PathPresence.CREATE = PathPresence(name="create")
PathPresence.ALLOW_MISSING = PathPresence(name="allow_missing")


class ManagedPath:
    """An absolute path that already passed ``role`` and ``presence``."""

    def __init__(self, *, path: Path, role: PathRole, presence: PathPresence) -> None:
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError(f"ManagedPath.path must be an absolute Path, got {path!r}")
        if not isinstance(role, PathRole):
            raise TypeError(f"role must be PathRole, got {type(role)!r}")
        if not isinstance(presence, PathPresence):
            raise TypeError(f"presence must be PathPresence, got {type(presence)!r}")
        self._path = path
        self._role = role
        self._presence = presence

    @property
    def path(self) -> Path:
        return self._path

    @property
    def role(self) -> PathRole:
        return self._role

    @property
    def presence(self) -> PathPresence:
        return self._presence

    def __fspath__(self) -> str:
        return str(self._path)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ManagedPath)
            and self._path == other._path
            and self._role == other._role
            and self._presence == other._presence
        )

    def __repr__(self) -> str:
        return (
            "ManagedPath("
            f"path={self._path!r}, role={self._role!r}, presence={self._presence!r})"
        )


def abs_path(path: Path | str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        raise ValueError(f"path must be absolute (caller-owned), got {resolved}")
    return resolved


def process_path(
    input_path: Path | str,
    *,
    role: PathRole,
    presence: PathPresence,
) -> ManagedPath:
    """Validate an absolute path against ``role`` and ``presence``.

    ``PathPresence.CREATE`` is directory-only: missing path is created.
    Relative input is an error. A relative path becomes absolute only with an
    explicit root (``join_under`` / ``resolve_under_project``), never cwd.
    """
    if role == PathRole.FILE and presence == PathPresence.CREATE:
        raise ValueError("PathPresence.CREATE is only valid for PathRole.DIRECTORY")
    path = abs_path(input_path)
    exists = path.exists()
    if exists:
        if role == PathRole.FILE and not path.is_file():
            raise ValueError(f"path must be a file, got directory {path}")
        if role == PathRole.DIRECTORY and not path.is_dir():
            raise ValueError(f"path must be a directory, got file {path}")
        return ManagedPath(path=path, role=role, presence=presence)
    if presence == PathPresence.MUST_EXIST:
        raise ValueError(f"{role.name} must exist: {path}")
    if presence == PathPresence.CREATE:
        path.mkdir(parents=True, exist_ok=True)
        return ManagedPath(path=path, role=role, presence=presence)
    return ManagedPath(path=path, role=role, presence=presence)
