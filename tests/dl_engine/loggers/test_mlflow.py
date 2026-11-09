import shutil
import tempfile
import unittest
from pathlib import Path

import pytest

from dl_toolbox.base.input_output.json_io import flatten_object
from dl_toolbox.dl_engine.loggers.logger import (
    LogExistence,
    LogStream,
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
    return logger._build_client()


def _remove_cwd_mlruns() -> None:
    mlruns = Path.cwd() / "mlruns"
    if mlruns.is_dir():
        shutil.rmtree(mlruns)


class MlflowLoggerTests(unittest.TestCase):
    def setUp(self) -> None:
        pytest.importorskip("mlflow.tracking")
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.addCleanup(_remove_cwd_mlruns)
        self.tmp_path = Path(tmp.name)

    def test_create_logs_and_close_leaves_running(self) -> None:
        logger = _logger(self.tmp_path)
        logger.setup(stream=LogStream.FIT, existence=LogExistence.CREATE)
        run_id = logger.run_id
        logger.log_params(
            flatten_object({"name": "dummy_conv", "model": {"name": "unet"}})
        )
        logger.log_tags({"stage": "train"})
        logger.log_scalars({"lr_finder/lr": 1e-3, "lr_finder/loss": 1.25}, step=0)
        logger.log_text("note", "curve", step=0)
        logger.log_table("rows", [{"loss": 1.25}], step=0)
        logger.close()
        client = _client(_logger(self.tmp_path))
        run = client.get_run(run_id)
        self.assertEqual(run.info.run_name, "dummy_conv")
        self.assertEqual(run.info.status, "RUNNING")
        self.assertEqual(run.data.params["name"], "dummy_conv")
        self.assertEqual(run.data.params["model.name"], "unet")
        self.assertEqual(run.data.tags["task"], "lr_finder")
        self.assertEqual(run.data.tags["stage"], "train")
        self.assertAlmostEqual(float(run.data.metrics["lr_finder/loss"]), 1.25)
        artifacts = client.list_artifacts(run_id)
        names = {item.path: True for item in artifacts}
        self.assertIn("note_step0.txt", names)
        self.assertIn("rows_step0.txt", names)

    def test_require_attaches_to_the_same_run(self) -> None:
        pytest.importorskip("numpy")
        import numpy as np

        fit = _logger(self.tmp_path)
        inspect = _logger(self.tmp_path)
        fit.setup(stream=LogStream.FIT, existence=LogExistence.CREATE)
        try:
            fit.log_scalars({"loss": 0.5}, step=1)
            inspect.setup(stream=LogStream.INSPECT, existence=LogExistence.REQUIRE)
            try:
                self.assertEqual(inspect.run_id, fit.run_id)
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
            self.assertEqual(run.info.status, "RUNNING")
        finally:
            run_id = fit.run_id
            fit.close()
        run = _client(_logger(self.tmp_path)).get_run(run_id)
        self.assertEqual(run.info.status, "RUNNING")
        artifacts = _client(_logger(self.tmp_path)).list_artifacts(
            run_id, path="inspection"
        )
        names = {item.path: True for item in artifacts}
        self.assertIn("inspection/note_step3.txt", names)
        self.assertIn("inspection/rows_step3.txt", names)

    def test_require_after_fit_close_still_logs(self) -> None:
        fit = _logger(self.tmp_path, tags={})
        inspect = _logger(self.tmp_path, tags={})
        fit.setup(stream=LogStream.FIT, existence=LogExistence.CREATE)
        fit.log_scalars({"loss": 0.25}, step=0)
        run_id = fit.run_id
        fit.close()
        inspect.setup(stream=LogStream.INSPECT, existence=LogExistence.REQUIRE)
        try:
            inspect.log_text("inspection/late", "after-fit", step=8)
            self.assertEqual(inspect.run_id, run_id)
        finally:
            inspect.close()
        run = _client(_logger(self.tmp_path)).get_run(run_id)
        self.assertEqual(run.info.status, "RUNNING")
        artifacts = _client(_logger(self.tmp_path)).list_artifacts(
            run_id, path="inspection"
        )
        names = {item.path: True for item in artifacts}
        self.assertIn("inspection/late_step8.txt", names)

    def test_create_when_run_exists_raises(self) -> None:
        first = _logger(self.tmp_path)
        first.setup(stream=LogStream.FIT, existence=LogExistence.CREATE)
        first.close()
        second = _logger(self.tmp_path)
        with self.assertRaisesRegex(LoggerError, "already exists"):
            second.setup(stream=LogStream.FIT, existence=LogExistence.CREATE)

    def test_require_before_create_raises(self) -> None:
        logger = _logger(self.tmp_path)
        with self.assertRaisesRegex(LoggerError, "does not exist"):
            logger.setup(stream=LogStream.INSPECT, existence=LogExistence.REQUIRE)

    def test_require_of_terminated_run_raises(self) -> None:
        fit = _logger(self.tmp_path)
        fit.setup(stream=LogStream.FIT, existence=LogExistence.CREATE)
        run_id = fit.run_id
        fit.close()
        client = _client(_logger(self.tmp_path))
        client.set_terminated(run_id)
        inspect = _logger(self.tmp_path)
        with self.assertRaisesRegex(LoggerError, "does not exist"):
            inspect.setup(stream=LogStream.INSPECT, existence=LogExistence.REQUIRE)

    def test_create_after_terminated_run_starts_a_new_run(self) -> None:
        first = _logger(self.tmp_path)
        first.setup(stream=LogStream.FIT, existence=LogExistence.CREATE)
        first_id = first.run_id
        first.close()
        client = _client(_logger(self.tmp_path))
        client.set_terminated(first_id, status="FAILED")
        second = _logger(self.tmp_path)
        second.setup(stream=LogStream.FIT, existence=LogExistence.CREATE)
        second_id = second.run_id
        try:
            self.assertNotEqual(second_id, first_id)
            second.log_scalars({"loss": 0.1}, step=0)
        finally:
            second.close()
        run = _client(_logger(self.tmp_path)).get_run(second_id)
        self.assertEqual(run.info.status, "RUNNING")
        self.assertEqual(
            _client(_logger(self.tmp_path)).get_run(first_id).info.status, "FAILED"
        )

    def test_log_before_setup_raises(self) -> None:
        logger = _logger(self.tmp_path, tags={})
        with self.assertRaisesRegex(LoggerError, "setup"):
            logger.log_params({"name": "dummy_conv"})
        logger.close()

    def test_log_artifact(self) -> None:
        logger = _logger(self.tmp_path, tags={})
        payload = self.tmp_path / "payload.txt"
        payload.write_text("hello", encoding="utf-8")
        logger.setup(stream=LogStream.FIT, existence=LogExistence.CREATE)
        try:
            logger.log_artifact(local_path=payload, artifact_path="extras")
            run_id = logger.run_id
        finally:
            logger.close()
        artifacts = _client(_logger(self.tmp_path)).list_artifacts(
            run_id, path="extras"
        )
        names = {item.path: True for item in artifacts}
        self.assertIn("extras/payload.txt", names)

    def test_log_model(self) -> None:
        pytest.importorskip("mlflow.pytorch")
        pytest.importorskip("torch")
        import torch

        logger = _logger(self.tmp_path, tags={})
        logger.setup(stream=LogStream.FIT, existence=LogExistence.CREATE)
        try:
            logger.log_model(model=torch.nn.Linear(2, 1), artifact_path="model")
            run_id = logger.run_id
        finally:
            logger.close()
        client = _client(_logger(self.tmp_path))
        run = client.get_run(run_id)
        outputs = run.outputs
        self.assertIsNotNone(outputs)
        self.assertEqual(len(outputs.model_outputs), 1)
        logged = client.get_logged_model(outputs.model_outputs[0].model_id)
        self.assertEqual(logged.name, "model")
