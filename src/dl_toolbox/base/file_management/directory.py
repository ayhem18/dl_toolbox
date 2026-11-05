"""``Directory`` is an absolute path whose role is directory.

Spec is what is true now (``missing`` | ``exists``). How it was obtained
(``must_exist`` / ``create`` / ``allow_missing``) is parse, not stored.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dl_toolbox.base.file_management.join import resolve_under_project
from dl_toolbox.base.file_management.parent_walk import find_named_directory
from dl_toolbox.base.file_management.paths import abs_path

if TYPE_CHECKING:
    from dl_toolbox.base.file_management.file import File


class DirectorySpec:
    """What is true of the directory **now**."""

    def __init__(self, *, name: str) -> None:
        if name not in ("exists", "missing"):
            raise ValueError(
                f"directory spec must be 'exists' or 'missing', got {name!r}"
            )
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DirectorySpec) and self._name == other._name

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"DirectorySpec.{self._name.upper()}"


DirectorySpec.EXISTS = DirectorySpec(name="exists")
DirectorySpec.MISSING = DirectorySpec(name="missing")


class DirectoryParsePolicy:
    """How a Directory is obtained. Not stored on the entity."""

    def __init__(self, *, name: str) -> None:
        if name not in ("must_exist", "create", "allow_missing"):
            raise ValueError(
                "directory parse policy must be 'must_exist', 'create', or "
                f"'allow_missing', got {name!r}"
            )
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DirectoryParsePolicy) and self._name == other._name

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"DirectoryParsePolicy.{self._name.upper()}"


DirectoryParsePolicy.MUST_EXIST = DirectoryParsePolicy(name="must_exist")
DirectoryParsePolicy.CREATE = DirectoryParsePolicy(name="create")
DirectoryParsePolicy.ALLOW_MISSING = DirectoryParsePolicy(name="allow_missing")


class Directory:
    """Absolute directory path plus ``DirectorySpec``. Construction validates."""

    def __init__(self, *, path: Path, spec: DirectorySpec) -> None:
        if not isinstance(path, Path):
            raise TypeError(f"path must be pathlib.Path, got {type(path)!r}")
        if not path.is_absolute():
            raise ValueError(f"path must be absolute (caller-owned), got {path}")
        if not isinstance(spec, DirectorySpec):
            raise TypeError(f"spec must be DirectorySpec, got {type(spec)!r}")
        exists = path.exists()
        if exists:
            if not path.is_dir():
                raise ValueError(f"path must be a directory, got file {path}")
            if spec != DirectorySpec.EXISTS:
                raise ValueError(
                    f"path exists as a directory but spec is {spec!r}: {path}"
                )
        elif spec != DirectorySpec.MISSING:
            raise ValueError(f"path does not exist but spec is {spec!r}: {path}")
        self._path = path
        self._spec = spec

    @property
    def path(self) -> Path:
        return self._path

    @property
    def spec(self) -> DirectorySpec:
        return self._spec

    @classmethod
    def from_absolute(
        cls, *, path: Path | str, policy: DirectoryParsePolicy
    ) -> Directory:
        """Absolute-path source. ``policy`` is not kept on the result."""
        return cls._from_absolute_path(path=abs_path(path), policy=policy)

    @classmethod
    def from_under(
        cls,
        *,
        root: Directory,
        relative_path: str,
        policy: DirectoryParsePolicy,
    ) -> Directory:
        """Root + posix-relative source. ``root`` must exist. No ``..``."""
        if not isinstance(root, Directory):
            raise TypeError(f"root must be Directory, got {type(root)!r}")
        if root.spec != DirectorySpec.EXISTS:
            raise ValueError(f"root Directory must exist: {root.path}")
        absolute = resolve_under_project(
            root.path, relative_path, label="relative_path"
        )
        return cls._from_absolute_path(path=absolute, policy=policy)

    @classmethod
    def from_parent_walk(
        cls, *, start: File | Directory, basename: str
    ) -> Directory:
        """Walk ancestors from ``start`` until ``basename`` is a directory."""
        from dl_toolbox.base.file_management.file import File

        if isinstance(start, File):
            walk_from = start.parent().path
        elif isinstance(start, Directory):
            if start.spec != DirectorySpec.EXISTS:
                raise ValueError(f"walk start Directory must exist: {start.path}")
            walk_from = start.path
        else:
            raise TypeError(f"start must be File or Directory, got {type(start)!r}")
        found = find_named_directory(basename, start=walk_from)
        return cls.from_absolute(path=found, policy=DirectoryParsePolicy.MUST_EXIST)

    @classmethod
    def _from_absolute_path(
        cls, *, path: Path, policy: DirectoryParsePolicy
    ) -> Directory:
        if not isinstance(policy, DirectoryParsePolicy):
            raise TypeError(
                f"policy must be DirectoryParsePolicy, got {type(policy)!r}"
            )
        exists = path.exists()
        if exists:
            if not path.is_dir():
                raise ValueError(f"path must be a directory, got file {path}")
            return cls(path=path, spec=DirectorySpec.EXISTS)
        if policy == DirectoryParsePolicy.MUST_EXIST:
            raise FileNotFoundError(f"directory must exist: {path}")
        if policy == DirectoryParsePolicy.CREATE:
            path.mkdir(parents=True, exist_ok=True)
            return cls(path=path, spec=DirectorySpec.EXISTS)
        return cls(path=path, spec=DirectorySpec.MISSING)

    def __fspath__(self) -> str:
        return str(self._path)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Directory)
            and self._path == other._path
            and self._spec == other._spec
        )

    def __hash__(self) -> int:
        return hash((self._path, self._spec))

    def __repr__(self) -> str:
        return f"Directory(path={self._path!r}, spec={self._spec!r})"
