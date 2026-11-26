"""In-memory ``RemoteStoreConnection`` for tests. Not collected by pytest.

Faithful to ``S3Connection``: keyed by ``(bucket, key)``; a directory push
writes the manifest last, a directory pull reads it and fetches exactly its
files and raises when it is absent.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from dl_toolbox.base.file_management.directory import Directory
from dl_toolbox.base.file_management.file import File
from dl_toolbox.remote_store.backends import RemoteBackend
from dl_toolbox.remote_store.connection import (
    DIRECTORY_MANIFEST_NAME,
    RemoteStoreConnection,
)
from dl_toolbox.remote_store.locations import parse_s3_location


class InMemoryConnection(RemoteStoreConnection):
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    @property
    def backend(self) -> str:
        return RemoteBackend.S3

    def exists(self, *, location: Mapping) -> bool:
        loc = parse_s3_location(location)
        return (loc.bucket, loc.path_within_bucket) in self.objects

    def push_file(self, *, location: Mapping, source: File) -> None:
        loc = parse_s3_location(location)
        self.objects[(loc.bucket, loc.path_within_bucket)] = source.path.read_bytes()

    def pull_file(
        self, *, location: Mapping, destination: Directory, name: str
    ) -> File:
        loc = parse_s3_location(location)
        data = self.objects[(loc.bucket, loc.path_within_bucket)]
        dest = destination.path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return File.from_absolute(path=dest)

    def push_directory(self, *, location: Mapping, source: Directory) -> None:
        loc = parse_s3_location(location)
        prefix = loc.path_within_bucket.rstrip("/") + "/"
        root = source.path
        files = sorted(
            p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
        )
        if DIRECTORY_MANIFEST_NAME in files:
            raise ValueError(f"{DIRECTORY_MANIFEST_NAME!r} is reserved")
        for rel in files:
            self.objects[(loc.bucket, prefix + rel)] = (root / rel).read_bytes()
        self.objects[(loc.bucket, prefix + DIRECTORY_MANIFEST_NAME)] = json.dumps(
            {"files": files}, indent=2, sort_keys=True
        ).encode("utf-8")

    def pull_directory(self, *, location: Mapping, destination: Directory) -> None:
        loc = parse_s3_location(location)
        prefix = loc.path_within_bucket.rstrip("/") + "/"
        manifest_key = (loc.bucket, prefix + DIRECTORY_MANIFEST_NAME)
        if manifest_key not in self.objects:
            raise FileNotFoundError(
                f"no completed directory at {loc.bucket}/{prefix}"
            )
        files = json.loads(self.objects[manifest_key].decode("utf-8"))["files"]
        for rel in files:
            out = destination.path / Path(PurePosixPath(rel))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(self.objects[(loc.bucket, prefix + rel)])
