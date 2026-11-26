"""``ProjectRemoteMirror``: the project-aware layer over a ``RemoteStoreConnection``.

It composes a connection (so MinIO/S3 vs Hugging Face is a config choice, not a
subclass), fixes the ``bucket`` at the project level, derives identity from the
file bytes through ``identity.hash_file``, and speaks only in paths relative to
``storage/``. It owns the mapping between the local tree and the remote.

Path handling is ``base.file_management``: ``resolve_under_project`` normalizes
and bounds a relative path under ``storage/`` (rejecting ``..`` and absolute
paths), and ``Directory.from_under`` / ``File`` carry presence. Nothing here
re-checks absoluteness or re-implements a relative-path guard.

Identity source asymmetry: ``push_file`` mints identity by hashing the local
bytes (they are in hand); ``object_exists`` / ``pull_file`` take the expected
hash from the identity document the caller already holds, and ``pull_file``
re-hashes the downloaded bytes and rejects a mismatch.

Note: building the ``location`` dict is the one backend-specific thing here (it
builds ``{bucket, path_within_bucket}``, the S3 shape). A location-builder hook
on the connection is the factoring to add when the Hugging Face backend lands.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
    DirectorySpec,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.file_management.join import resolve_under_project
from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.base.hashing.identity import hash_file, is_full_sha256
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.remote_store.connection import RemoteStoreConnection


class ProjectRemoteMirror:
    def __init__(
        self,
        *,
        connection: RemoteStoreConnection,
        hash_manager: HashManager,
        storage: Directory,
        bucket: str,
    ) -> None:
        if not isinstance(connection, RemoteStoreConnection):
            raise TypeError(
                f"connection must be RemoteStoreConnection, got {type(connection)!r}"
            )
        if not isinstance(hash_manager, HashManager):
            raise TypeError(
                f"hash_manager must be HashManager, got {type(hash_manager)!r}"
            )
        if not isinstance(storage, Directory):
            raise TypeError(f"storage must be Directory, got {type(storage)!r}")
        if storage.spec != DirectorySpec.EXISTS:
            raise ValueError(f"storage Directory must exist: {storage.path}")
        self._connection = connection
        self._hash_manager = hash_manager
        self._storage = storage
        self._bucket = parse_non_empty_str(bucket, label="bucket")

    @property
    def connection(self) -> RemoteStoreConnection:
        return self._connection

    @property
    def bucket(self) -> str:
        return self._bucket

    def push_file(self, *, local_file: File) -> str:
        """Overwrite the remote mirror of ``local_file``. Returns the identity."""
        if not isinstance(local_file, File):
            raise TypeError(f"local_file must be File, got {type(local_file)!r}")
        relative = self._relative_of(local_file)
        identity = hash_file(local_file.path)
        self._connection.push_file(
            identity=identity, location=self._location(relative), source=local_file
        )
        return identity

    def object_exists(self, *, relative_path: str, expected_hash: str) -> bool:
        identity = self._require_hash(expected_hash)
        relative = self._normalize(relative_path)
        return self._connection.object_exists(
            identity=identity, location=self._location(relative)
        )

    def pull_file(self, *, relative_path: str, expected_hash: str) -> File:
        """Overwrite the local file under ``storage/`` from the remote, then verify."""
        identity = self._require_hash(expected_hash)
        relative = self._normalize(relative_path)
        parts = PurePosixPath(relative)
        destination = self._destination_directory(parts.parent)
        pulled = self._connection.pull_file(
            identity=identity,
            location=self._location(relative),
            destination=destination,
            name=parts.name,
        )
        actual = hash_file(pulled.path)
        if actual != identity:
            raise ValueError(
                f"pulled {relative!r} hashes to {actual!r}, expected {identity!r}"
            )
        return pulled

    def push_directory(self, *, relative_path: str, identity: str) -> None:
        checked = self._require_hash(identity)
        relative = self._normalize(relative_path)
        source = Directory.from_under(
            root=self._storage,
            relative_path=relative,
            policy=DirectoryParsePolicy.MUST_EXIST,
        )
        self._connection.push_directory(
            identity=checked, location=self._location(relative), source=source
        )

    def pull_directory(self, *, relative_path: str, identity: str) -> None:
        checked = self._require_hash(identity)
        relative = self._normalize(relative_path)
        destination = Directory.from_under(
            root=self._storage,
            relative_path=relative,
            policy=DirectoryParsePolicy.CREATE,
        )
        self._connection.pull_directory(
            identity=checked, location=self._location(relative), destination=destination
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

    def _destination_directory(self, parent: PurePosixPath) -> Directory:
        if str(parent) in ("", "."):
            return self._storage
        return Directory.from_under(
            root=self._storage,
            relative_path=str(parent),
            policy=DirectoryParsePolicy.CREATE,
        )

    def _require_hash(self, raw: object) -> str:
        if not isinstance(raw, str) or not is_full_sha256(raw):
            raise ValueError(f"expected a 64-char lowercase SHA-256, got {raw!r}")
        return raw
