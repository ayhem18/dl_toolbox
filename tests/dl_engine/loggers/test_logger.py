"""Logger ABC: named constants, no setup defaults."""

from __future__ import annotations

import inspect

from dl_toolbox.dl_engine.loggers.logger import (
    EXISTENCE_CREATE,
    EXISTENCE_REQUIRE,
    STREAM_FIT,
    STREAM_INSPECT,
    Logger,
)
from dl_toolbox.dl_engine.loggers.mlflow import MlflowLogger
from dl_toolbox.dl_engine.loggers.tensorboard import TensorBoardLogger

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


def test_setup_has_no_defaults() -> None:
    params = inspect.signature(Logger.setup).parameters
    assert params["stream"].default is inspect.Parameter.empty
    assert params["existence"].default is inspect.Parameter.empty
    assert params["stream"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["existence"].kind is inspect.Parameter.KEYWORD_ONLY


def test_named_stream_and_existence_constants() -> None:
    assert STREAM_FIT == "fit"
    assert STREAM_INSPECT == "inspect"
    assert EXISTENCE_CREATE == "create"
    assert EXISTENCE_REQUIRE == "require"


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


def test_backends_implement_logger_methods() -> None:
    for cls in (TensorBoardLogger, MlflowLogger):
        for name in _METHODS:
            assert callable(getattr(cls, name))
            base = inspect.signature(getattr(Logger, name))
            impl = inspect.signature(getattr(cls, name))
            assert list(impl.parameters) == list(base.parameters)


def test_artifact_and_model_are_not_on_logger() -> None:
    assert not hasattr(Logger, "log_artifact")
    assert not hasattr(Logger, "log_model")
    assert callable(MlflowLogger.log_artifact)
    assert callable(MlflowLogger.log_model)
    assert not hasattr(TensorBoardLogger, "log_artifact")
    assert not hasattr(TensorBoardLogger, "log_model")
