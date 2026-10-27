from pathlib import Path

import pytest

from dl_toolbox.dl_engine.loggers.factory import get_logger
from dl_toolbox.dl_engine.loggers.placeholder.placeholder_manager import PlaceholderManager


def test_unknown_logger_name_raises(tmp_path: Path) -> None:
    manager = PlaceholderManager(log_dir=tmp_path / "run")
    with pytest.raises(ValueError, match="Unknown logger"):
        get_logger("not-a-backend", manager=manager)
