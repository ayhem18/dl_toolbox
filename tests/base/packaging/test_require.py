from __future__ import annotations

import importlib

import pytest

from dl_toolbox.base.packaging import Extra, IMPORT_HINTS, require


def test_unknown_package_is_rejected() -> None:
    with pytest.raises(ValueError, match="optional extra"):
        require(package="tqdm")


def test_package_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        require("sklearn")  # type: ignore[misc]


def test_map_covers_declared_imports() -> None:
    assert IMPORT_HINTS["torch"] == ("extra", Extra.DL)
    assert IMPORT_HINTS["torch.utils.tensorboard"] == ("extra", Extra.TENSORBOARD)
    assert IMPORT_HINTS["sklearn"] == ("dist", "scikit-learn")
    assert IMPORT_HINTS["cv2"] == ("extra", Extra.GEOMETRY)


def test_missing_extra_mentions_dl_toolbox_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing(name: str):
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", _missing)
    with pytest.raises(ImportError, match=r"dl-toolbox\[mlflow\]") as exc:
        require(package="mlflow")
    assert "uv add" in str(exc.value)
    assert "pip install" in str(exc.value)


def test_missing_dist_mentions_uv_add(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing(name: str):
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", _missing)
    with pytest.raises(ImportError, match=r"uv add scikit-learn"):
        require(package="sklearn")
