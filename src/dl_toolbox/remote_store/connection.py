"""``RemoteStoreConnection``: a live connection and every transport operation.

The remote is the source of truth. ``push_*`` overwrites the remote, ``pull_*``
overwrites the local file. ``object_exists`` is the resolve-or-fetch gate; it is
**file-only** and reads a single object key (the index entry), never a payload
tree. This ABC knows backends and nothing about dl_toolbox; the project mapping
is ``ProjectRemoteMirror``.

Local ends are ``File`` (an existing absolute file) and ``Directory`` (presence
carried on the object), so this layer does not re-check absoluteness or
existence: ``base.file_management`` already owns that.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping

from dl_toolbox.base.file_management.directory import Directory
from dl_toolbox.base.file_management.file import File


class RemoteStoreConnection(abc.ABC):
    @property
    @abc.abstractmethod
    def backend(self) -> str:
        """``RemoteBackend`` value this connection speaks."""

    @abc.abstractmethod
    def object_exists(self, *, identity: str, location: Mapping) -> bool:
        """Does the single object at ``(identity, location)`` exist. File-only."""

    @abc.abstractmethod
    def push_file(self, *, identity: str, location: Mapping, source: File) -> None:
        """Overwrite the remote object from ``source``, unconditionally."""

    @abc.abstractmethod
    def pull_file(
        self, *, identity: str, location: Mapping, destination: Directory, name: str
    ) -> File:
        """Overwrite ``name`` under ``destination`` from the remote object."""

    @abc.abstractmethod
    def push_directory(
        self, *, identity: str, location: Mapping, source: Directory
    ) -> None:
        """Upload a tree under the location prefix. No ``object_exists`` for a tree."""

    @abc.abstractmethod
    def pull_directory(
        self, *, identity: str, location: Mapping, destination: Directory
    ) -> None:
        """Download the tree under the location prefix into ``destination``."""
