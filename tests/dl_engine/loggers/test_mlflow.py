from pathlib import Path

import pytest

from dl_toolbox.base.input_output.json_io import flatten
from dl_toolbox.dl_engine.loggers.logger import (
    EXISTENCE_CREATE,
    EXISTENCE_REQUIRE,
    STREAM_FIT,
    STREAM_INSPECT,
    LoggerError,
)
from dl_toolbox.dl_engine.loggers.mlflow import MlflowLogger


def _logger(
    tmp_path: Path,
    *,
    run_name: str = "dummy_conv",
    tags: dict[str, str] | None = None,
) -> MlflowLogger:
    db = tmp_path / "mlflow" / "mlflow.sqlite"
    artifacts = tmp_path / "mlflow" / "artifacts"
    mapping = {"task": "lr_finder", "name": run_name} if tags is None else tags
    return MlflowLogger(
        backend_store_uri=f"sqlite:///{db.as_posix()}",
        artifact_store_uri=artifacts.as_posix(),
        experiment_name="lr_finder",
        run_name=run_name,
        tags=mapping,
    )


def _client(logger: MlflowLogger):
    return logger._client()


def test_create_logs_and_close_leaves_running(tmp_path: Path) -> None:
    pytest.importorskip("mlflow.tracking")
    logger = _logger(tmp_path)
    logger.setup(stream=STREAM_FIT, existence=EXISTENCE_CREATE)
    run_id = logger.run_id
    logger.log_params(flatten({"name": "dummy_conv", "model": {"name": "unet"}}))
    logger.log_tags({"stage": "train"})
    logger.log_scalars({"lr_finder/lr": 1e-3, "lr_finder/loss": 1.25}, step=0)
    logger.log_text("note", "curve", step=0)
    logger.log_table("rows", [{"loss": 1.25}], step=0)
    logger.close()
    client = _client(_logger(tmp_path))
    run = client.get_run(run_id)
    assert run.info.run_name == "dummy_conv"
    assert run.info.status == "RUNNING"
    assert run.data.params["name"] == "dummy_conv"
    assert run.data.params["model.name"] == "unet"
    assert run.data.tags["task"] == "lr_finder"
    assert run.data.tags["stage"] == "train"
    assert float(run.data.metrics["lr_finder/loss"]) == pytest.approx(1.25)
    artifacts = client.list_artifacts(run_id)
    names = {item.path: True for item in artifacts}
    assert "note_step0.txt" in names
    assert "rows_step0.txt" in names


def test_require_attaches_to_the_same_run(tmp_path: Path) -> None:
    pytest.importorskip("mlflow.tracking")
    pytest.importorskip("numpy")
    import numpy as np

    fit = _logger(tmp_path)
    inspect = _logger(tmp_path)
    fit.setup(stream=STREAM_FIT, existence=EXISTENCE_CREATE)
    try:
        fit.log_scalars({"loss": 0.5}, step=1)
        inspect.setup(stream=STREAM_INSPECT, existence=EXISTENCE_REQUIRE)
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
        run = _client(fit).get_run(fit.run_id)
        assert run.info.status == "RUNNING"
    finally:
        run_id = fit.run_id
        fit.close()
    run = _client(_logger(tmp_path)).get_run(run_id)
    assert run.info.status == "RUNNING"
    artifacts = _client(_logger(tmp_path)).list_artifacts(run_id, path="inspection")
    names = {item.path: True for item in artifacts}
    assert "inspection/note_step3.txt" in names
    assert "inspection/rows_step3.txt" in names


def test_require_after_fit_close_still_logs(tmp_path: Path) -> None:
    pytest.importorskip("mlflow.tracking")
    fit = _logger(tmp_path, tags={})
    inspect = _logger(tmp_path, tags={})
    fit.setup(stream=STREAM_FIT, existence=EXISTENCE_CREATE)
    fit.log_scalars({"loss": 0.25}, step=0)
    run_id = fit.run_id
    fit.close()
    inspect.setup(stream=STREAM_INSPECT, existence=EXISTENCE_REQUIRE)
    try:
        inspect.log_text("inspection/late", "after-fit", step=8)
        assert inspect.run_id == run_id
    finally:
        inspect.close()
    run = _client(_logger(tmp_path)).get_run(run_id)
    assert run.info.status == "RUNNING"
    artifacts = _client(_logger(tmp_path)).list_artifacts(run_id, path="inspection")
    names = {item.path: True for item in artifacts}
    assert "inspection/late_step8.txt" in names


def test_create_when_run_exists_raises(tmp_path: Path) -> None:
    pytest.importorskip("mlflow.tracking")
    first = _logger(tmp_path)
    first.setup(stream=STREAM_FIT, existence=EXISTENCE_CREATE)
    first.close()
    second = _logger(tmp_path)
    with pytest.raises(LoggerError, match="already exists"):
        second.setup(stream=STREAM_FIT, existence=EXISTENCE_CREATE)


def test_require_before_create_raises(tmp_path: Path) -> None:
    pytest.importorskip("mlflow.tracking")
    logger = _logger(tmp_path)
    with pytest.raises(LoggerError, match="does not exist"):
        logger.setup(stream=STREAM_INSPECT, existence=EXISTENCE_REQUIRE)


def test_require_of_terminated_run_raises(tmp_path: Path) -> None:
    pytest.importorskip("mlflow.tracking")
    fit = _logger(tmp_path)
    fit.setup(stream=STREAM_FIT, existence=EXISTENCE_CREATE)
    run_id = fit.run_id
    fit.close()
    client = _client(_logger(tmp_path))
    client.set_terminated(run_id)
    inspect = _logger(tmp_path)
    with pytest.raises(LoggerError, match="terminal"):
        inspect.setup(stream=STREAM_INSPECT, existence=EXISTENCE_REQUIRE)


def test_log_before_setup_raises(tmp_path: Path) -> None:
    pytest.importorskip("mlflow.tracking")
    logger = _logger(tmp_path, tags={})
    with pytest.raises(LoggerError, match="setup"):
        logger.log_params({"name": "dummy_conv"})
    logger.close()


def test_log_artifact(tmp_path: Path) -> None:
    pytest.importorskip("mlflow.tracking")
    logger = _logger(tmp_path, tags={})
    payload = tmp_path / "payload.txt"
    payload.write_text("hello", encoding="utf-8")
    logger.setup(stream=STREAM_FIT, existence=EXISTENCE_CREATE)
    try:
        logger.log_artifact(local_path=payload, artifact_path="extras")
        run_id = logger.run_id
    finally:
        logger.close()
    artifacts = _client(_logger(tmp_path)).list_artifacts(run_id, path="extras")
    names = {item.path: True for item in artifacts}
    assert "extras/payload.txt" in names


def test_log_model(tmp_path: Path) -> None:
    pytest.importorskip("mlflow.pytorch")
    pytest.importorskip("torch")
    import torch

    logger = _logger(tmp_path, tags={})
    logger.setup(stream=STREAM_FIT, existence=EXISTENCE_CREATE)
    try:
        logger.log_model(model=torch.nn.Linear(2, 1), artifact_path="model")
        run_id = logger.run_id
    finally:
        logger.close()
    client = _client(_logger(tmp_path))
    run = client.get_run(run_id)
    outputs = run.outputs
    assert outputs is not None
    assert len(outputs.model_outputs) == 1
    logged = client.get_logged_model(outputs.model_outputs[0].model_id)
    assert logged.name == "model"
