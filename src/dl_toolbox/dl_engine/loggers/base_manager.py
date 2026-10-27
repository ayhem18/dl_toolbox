"""Project-level logging store. Open-run state lives on Logger."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LoggingManager(ABC):
    @abstractmethod
    def prepare(self) -> None:
        """Create store directories and validate the backend. Idempotent."""
        raise NotImplementedError
