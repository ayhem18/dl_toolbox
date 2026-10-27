from pathlib import Path

import pytest

from dl_toolbox.dl_engine.loggers.base_logger import STREAM_INSPECT, LoggerError
from dl_toolbox.dl_engine.loggers.tensorboard.tensorboard_logger import TensorBoardLogger
from dl_toolbox.dl_engine.loggers.tensorboard.tensorboard_manager import TensorBoardManager


def _logger(tmp_path: Path) -> TensorBoardLogger:
    manager = TensorBoardManager(log_dir=tmp_path / "tb")
    return TensorBoardLogger(manager=manager, run_name="run")


def test_tensorboard_logger_writes_events(tmp_path: Path) -> None:
    pytest.importorskip("torch.utils.tensorboard")
    import torch

    logger = _logger(tmp_path)
    logger.setup()
    logger.log_scalars({"loss": 0.5}, step=1)
    logger.log_image("inspection/panel", torch.zeros(1, 8, 8), step=3)
    logger.log_table("inspection/rows", [{"loss": 0.5}], step=3)
    logger.log_text("note", "ok", step=3)
    logger.close()
    run_dir = tmp_path / "tb" / "run"
    events = list(run_dir.glob("events.out.tfevents.*"))
    assert len(events) >= 1


def test_inspect_stream_uses_suffix(tmp_path: Path) -> None:
    pytest.importorskip("torch.utils.tensorboard")
    import torch

    logger = _logger(tmp_path)
    logger.setup(stream=STREAM_INSPECT)
    logger.log_image("inspection/panel", torch.zeros(3, 4, 4), step=2)
    logger.close()
    names = [path.name for path in (tmp_path / "tb" / "run").glob("events.out.tfevents.*")]
    assert names
    n = len(names)
    matched = False
    for i in range(n):
        if names[i].endswith(".inspect"):
            matched = True
    assert matched


def test_tensorboard_log_before_setup_raises(tmp_path: Path) -> None:
    logger = _logger(tmp_path)
    with pytest.raises(LoggerError, match="setup"):
        logger.log_scalars({"loss": 1.0}, step=0)


def test_tensorboard_rejects_relative_log_dir() -> None:
    with pytest.raises(ValueError, match="absolute"):
        TensorBoardManager(log_dir=Path("relative/tb"))
