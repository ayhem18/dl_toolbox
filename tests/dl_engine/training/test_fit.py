from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from dl_toolbox.dl_engine.metrics.classification.classification import (
    binary_bce_loss_metric,
    binary_classification_accuracy_metric,
)
from dl_toolbox.dl_engine.metrics.contracts import KEY_TARGET
from dl_toolbox.dl_engine.metrics.metric_object import LEVEL_EPOCH, metric_log_name
from dl_toolbox.dl_engine.training.checkpoint import CheckpointManager
from dl_toolbox.dl_engine.training.config import (
    EarlyStoppingConfig,
    FitConfig,
    MonitorConfig,
    TrainerConfig,
)
from dl_toolbox.dl_engine.training.fit import build_metric_inputs, fit_classification
from dl_toolbox.dl_engine.loggers.factory import get_logger
from dl_toolbox.dl_engine.loggers.placeholder.placeholder_manager import PlaceholderManager


class _LinearBinary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _prepare_tensors(batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
    features, labels = batch
    return {"input": features, KEY_TARGET: labels}


def _forward(net: nn.Module, batch: dict[str, Tensor]) -> Tensor:
    return net(batch["input"])


def _loss(output: Tensor, batch: dict[str, Tensor]) -> Tensor:
    target = batch[KEY_TARGET].float().unsqueeze(1)
    return F.binary_cross_entropy_with_logits(output, target)


def test_fit_classification_lightning(tmp_path) -> None:
    x = torch.randn(32, 4)
    y = (x[:, 0] > 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    loss_metric = binary_bce_loss_metric()
    acc = binary_classification_accuracy_metric()
    specs = [(loss_metric, (), ("mean",)), (acc, (), ("acc",))]
    monitor = metric_log_name("loss", "val", LEVEL_EPOCH, "mean")
    monitor_config = MonitorConfig(key=monitor, mode="min")

    log_dir = tmp_path / "logs"
    run_dir = tmp_path / "run"
    logger = get_logger(
        "placeholder",
        manager=PlaceholderManager(log_dir=log_dir),
        run_name="test",
    )
    checkpoint_manager = CheckpointManager(
        save_top_k=-1,
        save_last=True,
        run_dir=run_dir,
        monitor=monitor_config,
    )

    model = _LinearBinary()
    trainer = fit_classification(
        model,
        loader,
        loader,
        torch.optim.Adam(model.parameters(), lr=0.05),
        prepare_tensors_fn=_prepare_tensors,
        forward_fn=_forward,
        loss_fn=_loss,
        build_metric_inputs_fn=build_metric_inputs,
        train_metrics=specs,
        val_metrics=specs,
        config=FitConfig(
            trainer=TrainerConfig(
                max_epochs=2,
                accelerator="cpu",
                devices=1,
                precision=32,
                enable_progress_bar=False,
            ),
            monitor=monitor_config,
            early_stopping=EarlyStoppingConfig(
                name=EarlyStoppingConfig.PATIENCE,
                patience=3,
                min_delta=1e-5,
            ),
        ),
        checkpoint_manager=checkpoint_manager,
        logger=logger,
        run_config={"monitor": monitor},
    )
    assert trainer.current_epoch >= 1
    checkpoints = run_dir / "checkpoints"
    assert (checkpoints / "catalog.json").is_file()
    ckpt_files = list(checkpoints.glob("epoch=*.ckpt"))
    assert ckpt_files
    assert (log_dir / "test" / "run_config.json").exists()
