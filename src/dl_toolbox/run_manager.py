"""Project contract: identity, runs folder, selected logging store.

This module has no filesystem layout. The opinionated tree lives in
``dl_toolbox.project_layout``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class RunManager(ABC):
    """What the rest of the toolbox needs from one project.

    Layout, logging JSON, and vendor store topology belong on a concrete
    class such as ``DefaultRunManager``.
    """

    @property
    @abstractmethod
    def project_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def project_runs_dir(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def tracking_backend(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def make_logging_manager(self) -> object:
        """Return the store manager for ``tracking_backend()``."""
        raise NotImplementedError
