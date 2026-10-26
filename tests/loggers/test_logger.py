"""Logger ABC helpers and NullLogger. No torch, no vendor SDK."""

from __future__ import annotations

import inspect

from dl_toolbox.loggers.base_logger import Logger, flatten_mapping
from dl_toolbox.loggers.mlflow.mlflow_logger import MlflowLogger
from dl_toolbox.loggers.placeholder.placeholder_logger import PlaceholderLogger
from dl_toolbox.loggers.tensorboard.tensorboard_logger import TensorBoardLogger
from dl_toolbox.loggers.wandb.wandb_logger import WandbLogger

from .null_logger import NullLogger

_METHODS = (
    "setup",
    "log_params",
    "log_tags",
    "log_scalars",
    "log_artifact",
    "close",
)


def test_flattens_nested_bools_lists_and_none() -> None:
    params = flatten_mapping(
        {
            "name": "dummy_conv",
            "model": {"name": "unet_resnet34", "params": {"pretrained": False}},
            "runtime": {"devices": 1, "workers": None},
            "tags": ["a", "b"],
        }
    )
    assert params["name"] == "dummy_conv"
    assert params["model.name"] == "unet_resnet34"
    assert params["model.params.pretrained"] == "false"
    assert params["runtime.devices"] == "1"
    assert params["runtime.workers"] == "null"
    assert params["tags"] == '["a", "b"]'


def test_null_logger_records_batched_scalars_and_close() -> None:
    logger = NullLogger()
    logger.setup()
    logger.log_params({"name": "dummy"})
    logger.log_tags({"task": "lr_finder"})
    logger.log_scalars({"lr_finder/lr": 0.1, "lr_finder/loss": 1.5}, step=0)
    logger.log_scalar("suggested/steepest", 0.01, step=0)
    logger.log_artifact("lr_curve.json")
    logger.close()
    logger.close()
    assert logger.setup_calls == 1
    assert logger.closed is True
    assert logger.params[0]["name"] == "dummy"
    assert logger.tags[0]["task"] == "lr_finder"
    assert logger.scalars[0][0]["lr_finder/lr"] == 0.1
    assert logger.scalars[0][0]["lr_finder/loss"] == 1.5
    assert logger.scalars[0][1] == 0
    assert logger.scalars[1][0]["suggested/steepest"] == 0.01
    assert logger.artifacts[0] == ("lr_curve.json", None)
    assert logger.run_id is None
    assert logger.tracking_uri is None


def test_log_scalar_step_is_keyword_only() -> None:
    params = inspect.signature(Logger.log_scalar).parameters
    assert params["step"].kind is inspect.Parameter.KEYWORD_ONLY


def test_log_scalars_step_is_keyword_only() -> None:
    params = inspect.signature(Logger.log_scalars).parameters
    assert params["step"].kind is inspect.Parameter.KEYWORD_ONLY


def test_backends_implement_logger_methods() -> None:
    for cls in (PlaceholderLogger, TensorBoardLogger, WandbLogger, MlflowLogger):
        for name in _METHODS:
            assert callable(getattr(cls, name))
            base = inspect.signature(getattr(Logger, name))
            impl = inspect.signature(getattr(cls, name))
            assert list(impl.parameters) == list(base.parameters)
