"""Optional-package imports with an install hint. Not vendor-specific."""

from __future__ import annotations

import importlib
from types import ModuleType


def require_package(package: str, *, install: str) -> ModuleType:
    try:
        return importlib.import_module(package)
    except ImportError as exc:
        raise ImportError(
            f"{package} is required and is not installed. "
            f"Install it in the project environment:\n  uv add {install}"
        ) from exc


def require_sklearn() -> ModuleType:
    return require_package("sklearn", install="scikit-learn")


def require_matplotlib() -> ModuleType:
    return require_package("matplotlib", install="matplotlib")
