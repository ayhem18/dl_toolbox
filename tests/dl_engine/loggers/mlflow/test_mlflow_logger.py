from pathlib import Path

import pytest

from dl_toolbox.dl_engine.loggers.base_logger import (
    STREAM_INSPECT,
    LoggerError,
    flatten_mapping,
)
from dl_toolbox.dl_engine.loggers.mlflow.mlflow_logger import MlflowLogger
from dl_toolbox.dl_engine.loggers.mlflow.mlflow_manager import MlflowManager


def _manager(tmp_path: Path) -> MlflowManager:
    return MlflowManager(
        db_path=tmp_path / "mlflow" / "mlflow.sqlite",
        artifact_root=tmp_path / "mlflow" / "artifacts",
        timeout_seconds=30,
    )


def _logger(tmp_path: Path, *, run_name: str = "dummy_conv") -> MlflowLogger:
    return MlflowLogger(
        manager=_manager(tmp_path),
        experiment="lr_finder",
        run_name=run_name,
        tags={"task": "lr_finder", "name": run_name},
    )


def test_setup_logs_and_close(tmp_path: Path) -> None:
    pytest.importorskip("mlflow.tracking")
    manager = _manager(tmp_path)
    logger = MlflowLogger(
        manager=manager,
        experiment="lr_finder",
        run_name="dummy_conv",
        tags={"task": "lr_finder", "name": "dummy_conv"},
    )
    try:
        logger.setup()
        assert logger.run_id is not None
        assert logger.tracking_uri is not None
        assert logger.tracking_uri.startswith("sqlite:///")
        logger.log_params(flatten_mapping({"name": "dummy_conv", "model": {"name": "unet"}}))
        logger.log_tags({"stage": "train"})
        logger.log_scalars({"lr_finder/lr": 1e-3, "lr_finder/loss": 1.25}, step=0)
        logger.log_text("note", "curve", step=0)
        logger.log_table("rows", [{"loss": 1.25}], step=0)
    finally:
        logger.close()
    run_id = logger.run_id
    assert run_id is not None
    client = manager.client()
    run = client.get_run(str(run_id))
    assert run.info.run_name == "dummy_conv"
    assert run.info.status == "FINISHED"
    assert run.data.params["name"] == "dummy_conv"
    assert run.data.params["model.name"] == "unet"
    assert run.data.tags["task"] == "lr_finder"
    assert run.data.tags["stage"] == "train"
    assert float(run.data.metrics["lr_finder/loss"]) == pytest.approx(1.25)
    artifacts = client.list_artifacts(str(run_id))
    names = {item.path: True for item in artifacts}
    assert "note_step0.txt" in names
    assert "rows_step0.txt" in names
    assert manager.db_path.is_file()
    pointer = manager.run_id_path(experiment="lr_finder", run_name="dummy_conv")
    assert pointer.is_file()
    assert manager.load_run_id(experiment="lr_finder", run_name="dummy_conv") == run_id


def test_inspect_attaches_without_ending_fit(tmp_path: Path) -> None:
    pytest.importorskip("mlflow.tracking")
    pytest.importorskip("numpy")
    import numpy as np

    manager = _manager(tmp_path)
    fit = MlflowLogger(
        manager=manager,
        experiment="lr_finder",
        run_name="dummy_conv",
        tags={"task": "lr_finder"},
    )
    inspect = MlflowLogger(
        manager=manager,
        experiment="lr_finder",
        run_name="dummy_conv",
        tags={"task": "lr_finder"},
    )
    fit.setup()
    try:
        fit.log_scalars({"loss": 0.5}, step=1)
        inspect.setup(stream=STREAM_INSPECT)
        try:
            assert inspect.run_id == fit.run_id
            inspect.log_image(
                "inspection/panel",
                np.zeros((4, 4, 3), dtype="float32"),
                step=3,
            )
            inspect.log_text("inspection/note", "ok", step=3)
            inspect.log_table("inspection/rows", [{"loss": 0.5}], step=3)
        finally:
            inspect.close()
        run = manager.client().get_run(str(fit.run_id))
        assert run.info.status == "RUNNING"
        experiment_id = manager.open_experiment(experiment="lr_finder")
        runs = manager.client().search_runs([experiment_id])
        assert len(runs) == 1
    finally:
        fit.close()
    run = manager.client().get_run(str(fit.run_id))
    assert run.info.status == "FINISHED"
    artifacts = manager.client().list_artifacts(str(fit.run_id), path="inspection")
    names = {item.path: True for item in artifacts}
    assert "inspection/note_step3.txt" in names
    assert "inspection/rows_step3.txt" in names


def test_inspect_after_fit_close_still_logs(tmp_path: Path) -> None:
    pytest.importorskip("mlflow.tracking")
    manager = _manager(tmp_path)
    fit = MlflowLogger(
        manager=manager,
        experiment="lr_finder",
        run_name="dummy_conv",
        tags={},
    )
    inspect = MlflowLogger(
        manager=manager,
        experiment="lr_finder",
        run_name="dummy_conv",
        tags={},
    )
    fit.setup()
    fit.log_scalars({"loss": 0.25}, step=0)
    fit.close()
    inspect.setup(stream=STREAM_INSPECT)
    try:
        inspect.log_text("inspection/late", "after-fit", step=8)
    finally:
        inspect.close()
    run = manager.client().get_run(str(fit.run_id))
    assert run.info.status == "FINISHED"
    artifacts = manager.client().list_artifacts(str(fit.run_id), path="inspection")
    names = {item.path: True for item in artifacts}
    assert "inspection/late_step8.txt" in names


def test_inspect_before_fit_raises(tmp_path: Path) -> None:
    pytest.importorskip("mlflow.tracking")
    logger = _logger(tmp_path)
    with pytest.raises(LoggerError, match="run_id recorded"):
        logger.setup(stream=STREAM_INSPECT)


def test_log_before_setup_raises(tmp_path: Path) -> None:
    pytest.importorskip("mlflow.tracking")
    logger = MlflowLogger(
        manager=_manager(tmp_path),
        experiment="lr_finder",
        run_name="dummy_conv",
        tags={},
    )
    with pytest.raises(LoggerError, match="setup"):
        logger.log_params({"name": "dummy_conv"})
    logger.close()


def test_manager_does_not_import_mlflow(tmp_path: Path) -> None:
    import sys

    names = [
        name
        for name in sys.modules
        if name == "mlflow" or name.startswith("mlflow.")
    ]
    n = len(names)
    for i in range(n):
        sys.modules.pop(names[i], None)
    manager = _manager(tmp_path)
    assert manager.db_path.name == "mlflow.sqlite"
    assert "mlflow" not in sys.modules
