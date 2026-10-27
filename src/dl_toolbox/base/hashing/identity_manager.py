"""Identity index contract. Default implementation is HashManager."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class IdentityManager(ABC):
    @abstractmethod
    def hash_payload(self, payload: object) -> str:
        raise NotImplementedError

    @abstractmethod
    def assert_name(self, name: object) -> str:
        raise NotImplementedError

    @abstractmethod
    def lookup(
        self,
        directory: Path,
        *,
        name: str,
        configuration_hash: str,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def claim(
        self,
        directory: Path,
        *,
        name: str,
        configuration_hash: str,
        extra: Mapping[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        raise NotImplementedError

    @abstractmethod
    def release_name(self, directory: Path, *, name: str) -> dict[str, Any] | None:
        raise NotImplementedError
