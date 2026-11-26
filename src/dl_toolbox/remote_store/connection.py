"""``RemoteStoreConnection``: backend-neutral byte transport, keyed only by
``location``.

The remote is the source of truth. ``push_*`` overwrites the remote; ``pull_*``
overwrites local. There is no hashing, no verification, and no identity at this
layer: a thing is addressed solely by its ``location`` (a backend-specific dict;
for S3 ``{bucket, path_within_bucket}``). Content addressing, where a project
wants it, lives in how the local paths are named, not here.

A directory is not a real object — it is a key prefix. ``push_directory`` writes
every file under the prefix and then, **strictly last**, a manifest
(``DIRECTORY_MANIFEST_NAME``) listing the files it wrote; its presence is what
marks the push complete. ``pull_directory`` reads that manifest and downloads
exactly the files it lists, so a half-finished push (manifest absent) is never
pulled as if complete. Per-file transfers are not mutually atomic — object
stores have no atomic multi-object swap — the manifest is the completeness
signal, not a transaction.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping

from dl_toolbox.base.file_management.directory import Directory
from dl_toolbox.base.file_management.file import File

DIRECTORY_MANIFEST_NAME = ".dltoolbox-manifest.json"


class RemoteStoreConnection(abc.ABC):
    @property
    @abc.abstractmethod
    def backend(self) -> str:
        """``RemoteBackend`` value this connection speaks."""

    @abc.abstractmethod
    def exists(self, *, location: Mapping) -> bool:
        """Is there a single object at ``location``. File-granular (a key)."""

    @abc.abstractmethod
    def push_file(self, *, location: Mapping, source: File) -> None:
        """Overwrite the remote object at ``location`` from ``source``."""

    @abc.abstractmethod
    def pull_file(
        self, *, location: Mapping, destination: Directory, name: str
    ) -> File:
        """Overwrite ``destination/name`` from the remote object at ``location``."""

    @abc.abstractmethod
    def push_directory(self, *, location: Mapping, source: Directory) -> None:
        """Overwrite every file under the ``location`` prefix, manifest written last."""

    @abc.abstractmethod
    def pull_directory(self, *, location: Mapping, destination: Directory) -> None:
        """Download the files the manifest lists under ``location`` into ``destination``."""
