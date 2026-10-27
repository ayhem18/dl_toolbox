from pathlib import Path

import pytest

from dl_toolbox.dl_engine.loggers.base_logger import LoggerError, flatten_mapping
from dl_toolbox.dl_engine.loggers.mlflow.mlflow_logger import MlflowLogger
from dl_toolbox.dl_engine.loggers.mlflow.mlflow_manager import MlflowManager


def _manager(tmp_path: Path) -> MlflowManager:
    return MlflowManager(
        db_path=tmp_path / "mlflow" / "mlflow.sqlite",
        artifact_root=tmp_path / "mlflow" / "artifacts",
        timeout_seconds=30,
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
    artifact = tmp_path / "note.txt"
    artifact.write_text("curve", encoding="utf-8")
    try:
        logger.setup()
        assert logger.run_id is not None
        assert logger.tracking_uri is not None
        assert logger.tracking_uri.startswith("sqlite:///")
        logger.log_params(flatten_mapping({"name": "dummy_conv", "model": {"name": "unet"}}))
        logger.log_tags({"stage": "train"})
        logger.log_scalars({"lr_finder/lr": 1e-3, "lr_finder/loss": 1.25}, step=0)
        logger.log_artifact(artifact)
    finally:
        logger.close()
    run_id = logger.run_id
    assert run_id is not None
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=manager.backend_store_uri)
    run = client.get_run(str(run_id))
    assert run.info.run_name == "dummy_conv"
    assert run.data.params["name"] == "dummy_conv"
    assert run.data.params["model.name"] == "unet"
    assert run.data.tags["task"] == "lr_finder"
    assert run.data.tags["stage"] == "train"
    assert float(run.data.metrics["lr_finder/loss"]) == pytest.approx(1.25)
    artifacts = client.list_artifacts(str(run_id))
    names = {item.path: True for item in artifacts}
    assert "note.txt" in names
    assert manager.db_path.is_file()


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
