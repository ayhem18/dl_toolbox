"""``S3Connection``: the boto3/MinIO implementation of ``RemoteStoreConnection``.

Constructed by ``S3Credentials.connect()`` with a live boto3 S3 client. boto3
and botocore are imported lazily, so importing this module never requires the
``s3`` extra; only building a client or calling an operation does.

Local ends are ``File`` / ``Directory``; destination keys resolve with
``resolve_under_project`` and nested output folders are created through
``Directory.from_under`` — the ``base.file_management`` helpers, not hand-rolled
path checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.file_management.join import resolve_under_project
from dl_toolbox.base.hashing.identity import is_full_sha256
from dl_toolbox.remote_store.backends import RemoteBackend
from dl_toolbox.remote_store.connection import RemoteStoreConnection
from dl_toolbox.remote_store.locations import S3Location, parse_s3_location

IDENTITY_METADATA_KEY = "dltoolbox-identity"


class S3Connection(RemoteStoreConnection):
    def __init__(self, *, client: object) -> None:
        self._client = client

    @property
    def backend(self) -> str:
        return RemoteBackend.S3

    def object_exists(self, *, identity: str, location: Mapping) -> bool:
        _require_identity(identity)
        loc = parse_s3_location(location)
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=loc.bucket, Key=_object_key(loc))
            return True
        except ClientError as exc:
            if _is_not_found(exc):
                return False
            raise

    def push_file(self, *, identity: str, location: Mapping, source: File) -> None:
        _require_identity(identity)
        loc = parse_s3_location(location)
        self._client.upload_file(
            Filename=str(source.path),
            Bucket=loc.bucket,
            Key=_object_key(loc),
            ExtraArgs={"Metadata": {IDENTITY_METADATA_KEY: identity}},
        )

    def pull_file(
        self, *, identity: str, location: Mapping, destination: Directory, name: str
    ) -> File:
        _require_identity(identity)
        loc = parse_s3_location(location)
        dest_path = resolve_under_project(destination.path, name, label="name")
        self._client.download_file(
            Bucket=loc.bucket, Key=_object_key(loc), Filename=str(dest_path)
        )
        return File.from_absolute(path=dest_path)

    def push_directory(
        self, *, identity: str, location: Mapping, source: Directory
    ) -> None:
        _require_identity(identity)
        loc = parse_s3_location(location)
        root = source.path
        prefix = _object_prefix(loc)
        files = sorted(p for p in root.rglob("*") if p.is_file())
        n_files = len(files)
        for i in range(n_files):
            relative = files[i].relative_to(root).as_posix()
            self._client.upload_file(
                Filename=str(files[i]),
                Bucket=loc.bucket,
                Key=prefix + relative,
                ExtraArgs={"Metadata": {IDENTITY_METADATA_KEY: identity}},
            )

    def pull_directory(
        self, *, identity: str, location: Mapping, destination: Directory
    ) -> None:
        _require_identity(identity)
        loc = parse_s3_location(location)
        prefix = _object_prefix(loc)
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=loc.bucket, Prefix=prefix):
            contents = page.get("Contents", [])
            n_objects = len(contents)
            for i in range(n_objects):
                key = contents[i]["Key"]
                if key.endswith("/") or not key.startswith(prefix):
                    continue
                relative = key[len(prefix):]
                if not relative:
                    continue
                self._prepare_subdirectory(destination, relative)
                dest_path = resolve_under_project(
                    destination.path, relative, label="key"
                )
                self._client.download_file(
                    Bucket=loc.bucket, Key=key, Filename=str(dest_path)
                )

    def _prepare_subdirectory(self, destination: Directory, relative: str) -> None:
        parent = PurePosixPath(relative).parent
        if str(parent) in ("", "."):
            return
        Directory.from_under(
            root=destination,
            relative_path=str(parent),
            policy=DirectoryParsePolicy.CREATE,
        )


def _object_key(location: S3Location) -> str:
    return location.path_within_bucket


def _object_prefix(location: S3Location) -> str:
    return location.path_within_bucket.rstrip("/") + "/"


def _require_identity(identity: object) -> str:
    if not isinstance(identity, str) or not is_full_sha256(identity):
        raise ValueError(
            f"identity must be a 64-char lowercase SHA-256, got {identity!r}"
        )
    return identity


def _is_not_found(exc: object) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, Mapping):
        return False
    code = response.get("Error", {}).get("Code")
    if code in {"404", "NoSuchKey", "NotFound"}:
        return True
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return status == 404
