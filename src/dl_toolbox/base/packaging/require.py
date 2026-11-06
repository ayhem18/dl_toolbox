"""Optional extras: import or error with an install hint."""

from __future__ import annotations

import importlib
from types import ModuleType


class Extra:
    DL = "dl"
    GEOMETRY = "geometry"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    MLFLOW = "mlflow"


class HintKind:
    EXTRA = "extra"
    DIST = "dist"


IMPORT_HINTS: dict[str, tuple[str, str]] = {
    "cv2": (HintKind.EXTRA, Extra.GEOMETRY),
    "lightning": (HintKind.EXTRA, Extra.DL),
    "matplotlib": (HintKind.DIST, "matplotlib"),
    "mlflow": (HintKind.EXTRA, Extra.MLFLOW),
    "sklearn": (HintKind.DIST, "scikit-learn"),
    "tensorboard": (HintKind.EXTRA, Extra.TENSORBOARD),
    "torch": (HintKind.EXTRA, Extra.DL),
    "torch.utils.tensorboard": (HintKind.EXTRA, Extra.TENSORBOARD),
    "wandb": (HintKind.EXTRA, Extra.WANDB),
}


def require(*, package: str) -> ModuleType:
    if not isinstance(package, str) or not package:
        raise ValueError(f"package must be a non-empty import name, got {package!r}")
    if package not in IMPORT_HINTS:
        raise ValueError(
            f"package {package!r} is not an optional extra this contract knows"
        )
    kind, hint = IMPORT_HINTS[package]
    try:
        return importlib.import_module(package)
    except ImportError as exc:
        if kind == HintKind.EXTRA:
            raise ImportError(
                f"{package} is required and is not installed.\n"
                f"  uv add 'dl-toolbox[{hint}]'\n"
                f"  pip install 'dl-toolbox[{hint}]'"
            ) from exc
        raise ImportError(
            f"{package} is required and is not installed.\n"
            f"  uv add {hint}"
        ) from exc
