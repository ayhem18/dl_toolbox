from pathlib import Path

import pytest

from dl_toolbox.dl_engine.loggers.logger import (
    EXISTENCE_CREATE,
    EXISTENCE_REQUIRE,
    STREAM_FIT,
    STREAM_INSPECT,
    LoggerError,
)
from dl_toolbox.dl_engine.loggers.tensorboard import TensorBoardLogger


def _logger(tmp_path: Path) -> TensorBoardLogger:
    return TensorBoardLogger(log_dir=tmp_path / "tb", run_name="run")


def test_create_writes_fit_events(tmp_path: Path) -> None:
    pytest.importorskip("torch.utils.tensorboard")
    import torch

    logger = _logger(tmp_path)
    logger.setup(stream=STREAM_FIT, existence=EXISTENCE_CREATE)
    logger.log_scalars({"loss": 0.5}, step=1)
    logger.log_image("inspection/panel", torch.zeros(1, 8, 8), step=3)
    logger.log_table("inspection/rows", [{"loss": 0.5}], step=3)
    logger.log_text("note", "ok", step=3)
    logger.close()
    run_dir = tmp_path / "tb" / "run"
    names = [path.name for path in run_dir.glob("events.out.tfevents.*")]
    assert names
    n = len(names)
    matched = False
    for i in range(n):
        if names[i].endswith(".fit"):
            matched = True
    assert matched


def test_inspect_require_uses_suffix(tmp_path: Path) -> None:
    pytest.importorskip("torch.utils.tensorboard")
    import torch

    fit = _logger(tmp_path)
    fit.setup(stream=STREAM_FIT, existence=EXISTENCE_CREATE)
    fit.close()
    inspect = TensorBoardLogger(log_dir=tmp_path / "tb", run_name="run")
    inspect.setup(stream=STREAM_INSPECT, existence=EXISTENCE_REQUIRE)
    inspect.log_image("inspection/panel", torch.zeros(3, 4, 4), step=2)
    inspect.close()
    names = [path.name for path in (tmp_path / "tb" / "run").glob("events.out.tfevents.*")]
    assert names
    n = len(names)
    matched = False
    for i in range(n):
        if names[i].endswith(".inspect"):
            matched = True
    assert matched


def test_require_without_run_dir_raises(tmp_path: Path) -> None:
    logger = _logger(tmp_path)
    with pytest.raises(LoggerError, match="does not exist"):
        logger.setup(stream=STREAM_INSPECT, existence=EXISTENCE_REQUIRE)
    assert not (tmp_path / "tb" / "run").exists()


def test_log_before_setup_raises(tmp_path: Path) -> None:
    logger = _logger(tmp_path)
    with pytest.raises(LoggerError, match="setup"):
        logger.log_scalars({"loss": 1.0}, step=0)


def test_setup_twice_raises(tmp_path: Path) -> None:
    pytest.importorskip("torch.utils.tensorboard")
    logger = _logger(tmp_path)
    logger.setup(stream=STREAM_FIT, existence=EXISTENCE_CREATE)
    with pytest.raises(LoggerError, match="already open"):
        logger.setup(stream=STREAM_FIT, existence=EXISTENCE_CREATE)
    logger.close()


def test_close_is_idempotent_and_setup_after_close_makes_a_new_file(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch.utils.tensorboard")
    logger = _logger(tmp_path)
    logger.setup(stream=STREAM_FIT, existence=EXISTENCE_CREATE)
    logger.close()
    logger.close()
    logger.setup(stream=STREAM_FIT, existence=EXISTENCE_CREATE)
    logger.close()
    events = list((tmp_path / "tb" / "run").glob("events.out.tfevents.*"))
    assert len(events) == 2


def test_relative_log_dir_raises_without_touching_disk(tmp_path: Path) -> None:
    relative = Path("relative/tb")
    with pytest.raises(ValueError, match="absolute"):
        TensorBoardLogger(log_dir=relative, run_name="run")
    assert not relative.exists()
    assert not (tmp_path / "relative").exists()
