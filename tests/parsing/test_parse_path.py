from pathlib import Path

import pytest

from dl_toolbox.base.parsing.parse_path import parse_absolute_path


def test_absolute_path(tmp_path: Path) -> None:
    parsed = parse_absolute_path(str(tmp_path / "db.sqlite"), label="path")
    assert parsed == tmp_path / "db.sqlite"


def test_relative_path_fails() -> None:
    with pytest.raises(ValueError, match="absolute"):
        parse_absolute_path("mlflow/db.sqlite", label="path")


def test_empty_fails() -> None:
    with pytest.raises(ValueError, match="path"):
        parse_absolute_path("  ", label="path")
