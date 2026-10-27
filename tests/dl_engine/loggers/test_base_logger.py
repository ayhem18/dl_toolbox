"""Logger ABC helpers and NullLogger. No torch, no vendor SDK."""

from __future__ import annotations

import inspect

from dl_toolbox.dl_engine.loggers.base_logger import (
    STREAM_FIT,
    STREAM_INSPECT,
    Logger,
    flatten_mapping,
)
from dl_toolbox.dl_engine.loggers.mlflow.mlflow_logger import MlflowLogger
from dl_toolbox.dl_engine.loggers.placeholder.placeholder_logger import PlaceholderLogger
from dl_toolbox.dl_engine.loggers.tensorboard.tensorboard_logger import TensorBoardLogger
from dl_toolbox.dl_engine.loggers.wandb.wandb_logger import WandbLogger

from .null_logger import NullLogger

_METHODS = (
    "setup",
    "log_params",
    "log_tags",
    "log_scalars",
    "log_image",
    "log_table",
    "log_text",
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
    logger.log_text("lr_curve", "{}", step=0)
    logger.log_image("panel", object(), step=3)
    logger.log_table("rows", [{"loss": 0.1}], step=3)
    logger.close()
    logger.close()
    assert logger.setup_calls == 1
    assert logger.setup_stream == STREAM_FIT
    assert logger.closed is True
    assert logger.params[0]["name"] == "dummy"
    assert logger.tags[0]["task"] == "lr_finder"
    assert logger.scalars[0][0]["lr_finder/lr"] == 0.1
    assert logger.scalars[0][0]["lr_finder/loss"] == 1.5
    assert logger.scalars[0][1] == 0
    assert logger.scalars[1][0]["suggested/steepest"] == 0.01
    assert logger.texts[0] == ("lr_curve", "{}", 0)
    assert logger.images[0][0] == "panel"
    assert logger.images[0][2] == 3
    assert logger.tables[0][0] == "rows"
    assert logger.run_id is None
    assert logger.tracking_uri is None


def test_log_scalar_step_is_keyword_only() -> None:
    params = inspect.signature(Logger.log_scalar).parameters
    assert params["step"].kind is inspect.Parameter.KEYWORD_ONLY


def test_log_scalars_step_is_keyword_only() -> None:
    params = inspect.signature(Logger.log_scalars).parameters
    assert params["step"].kind is inspect.Parameter.KEYWORD_ONLY


def test_image_table_text_step_is_keyword_only() -> None:
    for name in ("log_image", "log_table", "log_text"):
        params = inspect.signature(getattr(Logger, name)).parameters
        assert params["step"].kind is inspect.Parameter.KEYWORD_ONLY


def test_setup_stream_default_is_fit() -> None:
    params = inspect.signature(Logger.setup).parameters
    assert params["stream"].default == STREAM_FIT
    assert STREAM_INSPECT == "inspect"


def test_backends_implement_logger_methods() -> None:
    for cls in (PlaceholderLogger, TensorBoardLogger, WandbLogger, MlflowLogger):
        for name in _METHODS:
            assert callable(getattr(cls, name))
            base = inspect.signature(getattr(Logger, name))
            impl = inspect.signature(getattr(cls, name))
            assert list(impl.parameters) == list(base.parameters)
