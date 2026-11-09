# PARKED: no contract yet. Not collected by pytest until this package is rewritten.
from __future__ import annotations

import pytest

pytest.skip("Parked: this package has no contract yet.", allow_module_level=True)

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR

from dl_toolbox.dl_engine._loggers.base_logger import STREAM_FIT, Logger
from dl_toolbox.dl_engine.training.module import (
    ClassificationLightningModule,
    _TRACK_PROG_BAR,
    _TRACK_SILENT,
)


class _RecordingLogger(Logger):
    def __init__(self) -> None:
        self.setup_calls = 0
        self.close_calls = 0
        self.params: list[Mapping[str, str]] = []
        self.texts: list[tuple[str, str, int]] = []
        self.scalars: list[tuple[dict[str, float], int]] = []

    def setup(self, *, stream: str = STREAM_FIT) -> None:
        del stream
        self.setup_calls += 1

    def log_params(self, params: Mapping[str, str]) -> None:
        self.params.append(dict(params))

    def log_tags(self, tags: Mapping[str, str]) -> None:
        del tags

    def log_scalars(self, values: Mapping[str, float], *, step: int) -> None:
        self.scalars.append((dict(values), int(step)))

    def log_image(self, tag: str, image: object, *, step: int) -> None:
        del tag, image, step

    def log_table(
        self, tag: str, rows: Sequence[Mapping[str, Any]], *, step: int
    ) -> None:
        del tag, rows, step

    def log_text(self, tag: str, text: str, *, step: int) -> None:
        self.texts.append((tag, text, int(step)))

    def close(self) -> None:
        self.close_calls += 1

    def cleanup(self) -> None:
        self.close()


class _FakeTrainer:
    def __init__(
        self,
        *,
        is_global_zero: bool,
        epoch: int = 0,
        callback_metrics: dict[str, Tensor] | None = None,
    ) -> None:
        self.is_global_zero = is_global_zero
        self.current_epoch = epoch
        self.callback_metrics = callback_metrics or {}
        self.sanity_checking = False
        self.global_rank = 0 if is_global_zero else 1
        self.world_size = 1 if is_global_zero else 4


class _CpuModule(ClassificationLightningModule):
    @property
    def device(self) -> torch.device:
        return torch.device("cpu")


def _identity_prepare(raw_batch: Any) -> Mapping[str, Any]:
    return raw_batch


def _forward(net: nn.Module, batch: Mapping[str, Any]) -> Tensor:
    del net, batch
    return torch.zeros(1)


def _loss(output: Tensor, batch: Mapping[str, Any]) -> Tensor:
    del batch
    return output.sum()


def _metric_inputs(output: Tensor, batch: Mapping[str, Any]) -> Mapping[str, Tensor]:
    del batch
    return {"output": output}


def _module(
    logger: Logger | None = None,
    *,
    lr_scheduler_factory=None,
) -> _CpuModule:
    net = nn.Linear(2, 1)
    return _CpuModule(
        net=net,
        optimizer=torch.optim.SGD(net.parameters(), lr=0.1),
        epoch_logger=logger or _RecordingLogger(),
        prepare_tensors_fn=_identity_prepare,
        forward_fn=_forward,
        loss_fn=_loss,
        build_metric_inputs_fn=_metric_inputs,
        train_metrics=[],
        val_metrics=[],
        monitor_key="loss/val/epoch/mean",
        run_config={"task": "unit"},
        run_config_filename="run_config",
        lr_scheduler_factory=lr_scheduler_factory,
    )


def test_epoch_track_flags_sync_dist_and_not_per_step() -> None:
    for track in (_TRACK_SILENT, _TRACK_PROG_BAR):
        assert track["sync_dist"] is True
        assert track["on_step"] is False
        assert track["on_epoch"] is True
        assert track["logger"] is False


def test_log_epoch_row_logs_expected_keys_when_rank_has_no_items() -> None:
    module = _module()
    module._trainer = _FakeTrainer(is_global_zero=True, epoch=0)
    logged: list[str] = []

    def _log(name: str, value: float, **kwargs: Any) -> None:
        del kwargs
        logged.append(name)
        assert value == 0.0

    module.log = _log  # type: ignore[method-assign]
    module._log_epoch_row(
        {},
        track=_TRACK_PROG_BAR,
        expected_keys=["loss/val/epoch/mean"],
    )
    assert logged == ["loss/val/epoch/mean"]


def test_epoch_end_self_log_uses_sync_dist() -> None:
    module = _module()
    module._trainer = _FakeTrainer(is_global_zero=True, epoch=4)
    logged: list[dict[str, Any]] = []

    def _log(name: str, value: float, **kwargs: Any) -> None:
        del name, value
        logged.append(kwargs)

    module.log = _log  # type: ignore[method-assign]
    module._log_epoch_row(
        {"loss/train/epoch/mean": 0.5},
        track=_TRACK_SILENT,
    )
    assert logged
    assert logged[0]["sync_dist"] is True
    assert logged[0]["on_step"] is False


def test_toolbox_logger_writes_only_on_rank_zero() -> None:
    logger = _RecordingLogger()
    module = _module(logger)
    module._trainer = _FakeTrainer(is_global_zero=False, epoch=2)
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    module._log_epoch_row(
        {"loss/val/epoch/mean": 1.25},
        track=_TRACK_PROG_BAR,
    )
    assert logger.scalars == []


def test_toolbox_logger_writes_synced_row_on_rank_zero() -> None:
    logger = _RecordingLogger()
    module = _module(logger)
    synced = torch.tensor(0.75)
    module._trainer = _FakeTrainer(
        is_global_zero=True,
        epoch=3,
        callback_metrics={"loss/val/epoch/mean": synced},
    )
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    module._log_epoch_row(
        {"loss/val/epoch/mean": 1.25},
        track=_TRACK_PROG_BAR,
    )
    assert logger.scalars == [({"loss/val/epoch/mean": 0.75}, 3)]


def test_on_fit_start_and_end_are_rank_zero_only() -> None:
    logger = _RecordingLogger()
    module = _module(logger)
    module._trainer = _FakeTrainer(is_global_zero=False)
    module.on_fit_start()
    module.on_fit_end()
    assert logger.setup_calls == 0
    assert logger.close_calls == 0
    assert logger.params == []

    rank0 = _RecordingLogger()
    module0 = _module(rank0)
    module0._trainer = _FakeTrainer(is_global_zero=True)
    module0.on_fit_start()
    module0.on_fit_end()
    assert rank0.setup_calls == 1
    assert rank0.close_calls == 1
    assert rank0.params
    assert rank0.texts[0][0] == "run_config"


def test_prepare_moves_tensors_and_leaves_strings() -> None:
    module = _module()
    image = torch.ones(1, 1, 4, 8)
    batch = {
        "image": image,
        "mask": torch.zeros(1, 1, 4, 8),
        "dcm_file_path": ["scan.dcm"],
        "frame_number": ["7"],
        "diameter_label": ["large"],
    }
    prepared = module._prepare(batch)
    assert prepared["dcm_file_path"] is batch["dcm_file_path"]
    assert prepared["frame_number"] is batch["frame_number"]
    assert prepared["diameter_label"] is batch["diameter_label"]
    assert isinstance(prepared["image"], Tensor)
    assert prepared["image"].device.type == "cpu"
    torch.testing.assert_close(prepared["image"], image)


def test_configure_optimizers_returns_epoch_scheduler() -> None:
    def factory(optimizer):
        return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    module = _module(lr_scheduler_factory=factory)
    configured = module.configure_optimizers()
    assert isinstance(configured, dict)
    assert configured["optimizer"] is module._optimizer
    assert configured["lr_scheduler"]["interval"] == "epoch"
    assert isinstance(configured["lr_scheduler"]["scheduler"], LambdaLR)


def test_configure_optimizers_without_scheduler_is_optimizer_only() -> None:
    module = _module()
    assert module.configure_optimizers() is module._optimizer
