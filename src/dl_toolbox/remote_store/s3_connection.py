"""``S3Connection``: the boto3/MinIO implementation of ``RemoteStoreConnection``.

Pure, location-keyed byte transport. boto3/botocore are imported lazily, so
importing this module never requires the ``s3`` extra; only building a client
or calling an operation does. A directory push writes the manifest
(``DIRECTORY_MANIFEST_NAME``) last; a directory pull reads it and fetches
exactly the files it lists.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.file_management.join import resolve_under_project
from dl_toolbox.remote_store.backends import RemoteBackend
from dl_toolbox.remote_store.connection import (
    DIRECTORY_MANIFEST_NAME,
    RemoteStoreConnection,
)
from dl_toolbox.remote_store.locations import S3Location, parse_s3_location


class S3Connection(RemoteStoreConnection):
    def __init__(self, *, client: object) -> None:
        self._client = client

    @property
    def backend(self) -> str:
        return RemoteBackend.S3

    def exists(self, *, location: Mapping) -> bool:
        loc = parse_s3_location(location)
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=loc.bucket, Key=_object_key(loc))
            return True
        except ClientError as exc:
            if _is_not_found(exc):
                return False
            raise

    def push_file(self, *, location: Mapping, source: File) -> None:
        loc = parse_s3_location(location)
        self._client.upload_file(
            Filename=str(source.path), Bucket=loc.bucket, Key=_object_key(loc)
        )

    def pull_file(
        self, *, location: Mapping, destination: Directory, name: str
    ) -> File:
        loc = parse_s3_location(location)
        dest_path = resolve_under_project(destination.path, name, label="name")
        self._client.download_file(
            Bucket=loc.bucket, Key=_object_key(loc), Filename=str(dest_path)
        )
        return File.from_absolute(path=dest_path)

    def push_directory(self, *, location: Mapping, source: Directory) -> None:
        loc = parse_s3_location(location)
        prefix = _object_prefix(loc)
        root = source.path
        files = sorted(
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file()
        )
        if DIRECTORY_MANIFEST_NAME in files:
            raise ValueError(
                f"{DIRECTORY_MANIFEST_NAME!r} is reserved and cannot be a member "
                f"of a pushed directory"
            )
        n_files = len(files)
        for i in range(n_files):
            self._client.upload_file(
                Filename=str(root / files[i]),
                Bucket=loc.bucket,
                Key=prefix + files[i],
            )
        # Manifest strictly last: its presence marks the push complete.
        self._client.put_object(
            Bucket=loc.bucket,
            Key=prefix + DIRECTORY_MANIFEST_NAME,
            Body=_manifest_bytes(files),
            ContentType="application/json",
        )

    def pull_directory(self, *, location: Mapping, destination: Directory) -> None:
        loc = parse_s3_location(location)
        prefix = _object_prefix(loc)
        files = self._read_manifest(loc.bucket, prefix)
        n_files = len(files)
        for i in range(n_files):
            relative = files[i]
            _ensure_subdirectory(destination, relative)
            dest_path = resolve_under_project(
                destination.path, relative, label="manifest entry"
            )
            self._client.download_file(
                Bucket=loc.bucket, Key=prefix + relative, Filename=str(dest_path)
            )

    def _read_manifest(self, bucket: str, prefix: str) -> list[str]:
        from botocore.exceptions import ClientError

        try:
            response = self._client.get_object(
                Bucket=bucket, Key=prefix + DIRECTORY_MANIFEST_NAME
            )
        except ClientError as exc:
            if _is_not_found(exc):
                raise FileNotFoundError(
                    f"no completed directory at {bucket}/{prefix} "
                    f"(manifest {DIRECTORY_MANIFEST_NAME} absent)"
                ) from exc
            raise
        return _parse_manifest(response["Body"].read())


def _object_key(location: S3Location) -> str:
    return location.path_within_bucket


def _object_prefix(location: S3Location) -> str:
    return location.path_within_bucket.rstrip("/") + "/"


def _manifest_bytes(files: list[str]) -> bytes:
    return json.dumps({"files": files}, indent=2, sort_keys=True).encode("utf-8")


def _parse_manifest(raw: bytes) -> list[str]:
    document = json.loads(raw.decode("utf-8"))
    if not isinstance(document, dict) or "files" not in document:
        raise ValueError("directory manifest must be a JSON object with 'files'")
    files = document["files"]
    if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
        raise ValueError("directory manifest 'files' must be a list of strings")
    return files


def _ensure_subdirectory(destination: Directory, relative: str) -> None:
    from pathlib import PurePosixPath

    parent = PurePosixPath(relative).parent
    if str(parent) in ("", "."):
        return
    Directory.from_under(
        root=destination,
        relative_path=str(parent),
        policy=DirectoryParsePolicy.CREATE,
    )


def _is_not_found(exc: object) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, Mapping):
        return False
    code = response.get("Error", {}).get("Code")
    if code in {"404", "NoSuchKey", "NotFound"}:
        return True
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return status == 404
