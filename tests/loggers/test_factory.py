from pathlib import Path

import pytest

from dl_toolbox.loggers.factory import get_logger
from dl_toolbox.loggers.placeholder.default_configurations import (
    default_placeholder_setup,
)
from dl_toolbox.loggers.placeholder.placeholder_manager import PlaceholderManager


def test_unknown_logger_name_raises(tmp_path: Path) -> None:
    manager = PlaceholderManager(default_placeholder_setup(log_dir=tmp_path / "run"))
    with pytest.raises(ValueError, match="Unknown logger"):
        get_logger("not-a-backend", manager=manager)
