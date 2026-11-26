"""``ProjectRemoteMirror``: the default layout over a ``RemoteStoreConnection``.

It composes a connection, fixes the ``bucket`` at the project level, and speaks
only in paths relative to ``storage/``. Its whole job is to turn a
``storage/``-relative path into a ``location`` and delegate: push the local
version up, pull the remote truth down, overwriting. No hashing, no
verification, no identity — content addressing, where a project wants it, is in
how the local paths under ``storage/`` are named, and this mirror reflects them
verbatim.

Building the ``location`` dict (``{bucket, path_within_bucket}``, the S3 shape)
is the one backend-specific line here; a location hook on the connection is the
factoring to add when a second backend lands.
"""

from __future__ import annotations

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
    DirectorySpec,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.file_management.join import resolve_under_project
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.remote_store.connection import RemoteStoreConnection


class ProjectRemoteMirror:
    def __init__(
        self,
        *,
        connection: RemoteStoreConnection,
        storage: Directory,
        bucket: str,
    ) -> None:
        if not isinstance(connection, RemoteStoreConnection):
            raise TypeError(
                f"connection must be RemoteStoreConnection, got {type(connection)!r}"
            )
        if not isinstance(storage, Directory):
            raise TypeError(f"storage must be Directory, got {type(storage)!r}")
        if storage.spec != DirectorySpec.EXISTS:
            raise ValueError(f"storage Directory must exist: {storage.path}")
        self._connection = connection
        self._storage = storage
        self._bucket = parse_non_empty_str(bucket, label="bucket")

    @property
    def connection(self) -> RemoteStoreConnection:
        return self._connection

    @property
    def bucket(self) -> str:
        return self._bucket

    def push_file(self, *, local_file: File) -> str:
        """Overwrite the remote mirror of ``local_file``. Returns its relative path."""
        if not isinstance(local_file, File):
            raise TypeError(f"local_file must be File, got {type(local_file)!r}")
        relative = self._relative_of(local_file)
        self._connection.push_file(
            location=self._location(relative), source=local_file
        )
        return relative

    def pull_file(self, *, relative_path: str) -> File:
        """Overwrite the local file under ``storage/`` from the remote truth."""
        relative = self._normalize(relative_path)
        parts = relative.rsplit("/", 1)
        if len(parts) == 1:
            destination, name = self._storage, parts[0]
        else:
            destination = Directory.from_under(
                root=self._storage,
                relative_path=parts[0],
                policy=DirectoryParsePolicy.CREATE,
            )
            name = parts[1]
        return self._connection.pull_file(
            location=self._location(relative), destination=destination, name=name
        )

    def exists(self, *, relative_path: str) -> bool:
        relative = self._normalize(relative_path)
        return self._connection.exists(location=self._location(relative))

    def push_directory(self, *, relative_path: str) -> None:
        relative = self._normalize(relative_path)
        source = Directory.from_under(
            root=self._storage,
            relative_path=relative,
            policy=DirectoryParsePolicy.MUST_EXIST,
        )
        self._connection.push_directory(
            location=self._location(relative), source=source
        )

    def pull_directory(self, *, relative_path: str) -> None:
        relative = self._normalize(relative_path)
        destination = Directory.from_under(
            root=self._storage,
            relative_path=relative,
            policy=DirectoryParsePolicy.CREATE,
        )
        self._connection.pull_directory(
            location=self._location(relative), destination=destination
        )

    def _location(self, relative: str) -> dict[str, str]:
        return {"bucket": self._bucket, "path_within_bucket": relative}

    def _normalize(self, relative_path: object) -> str:
        absolute = resolve_under_project(
            self._storage.path, relative_path, label="relative_path"
        )
        return absolute.relative_to(self._storage.path.resolve()).as_posix()

    def _relative_of(self, local_file: File) -> str:
        try:
            relative = local_file.path.resolve().relative_to(
                self._storage.path.resolve()
            )
        except ValueError as exc:
            raise ValueError(
                f"{local_file.path} is not under storage {self._storage.path}"
            ) from exc
        return self._normalize(relative.as_posix())
