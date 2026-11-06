from pathlib import Path

import pytest

from dl_toolbox.dl_engine._loggers.factory import get_logger
from dl_toolbox.dl_engine._loggers.tensorboard.tensorboard_manager import TensorBoardManager


def test_unknown_logger_name_raises(tmp_path: Path) -> None:
    manager = TensorBoardManager(log_dir=tmp_path / "run")
    with pytest.raises(ValueError, match="Unknown logger"):
        get_logger("not-a-backend", manager=manager)
