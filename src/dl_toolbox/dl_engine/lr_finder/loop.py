"""Range-test setup, restore, and the exponential LR sweep."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dl_toolbox.dl_engine.lr_finder.batches import (
    cycling_batches,
    prepare_on_device,
    set_optimizer_lr,
)
from dl_toolbox.dl_engine.lr_finder.config import LrFinderConfig
from dl_toolbox.dl_engine.lr_finder.errors import LrFinderError
from dl_toolbox.dl_engine.lr_finder.restore import restore_trainable, snapshot_trainable
from dl_toolbox.dl_engine.lr_finder.runtime import resolve_device
from dl_toolbox.dl_engine.lr_finder.spec import LrFinderSpec, validate_lr_finder_spec
from dl_toolbox.dl_engine.lr_finder.step import LrFinderStep
from dl_toolbox.dl_engine_utils.precision import AmpSession
from dl_toolbox.dl_engine_utils.progress import range_progress


@dataclass
class RangeTestInput:
    model: nn.Module
    optimizer: Optimizer
    train_loader: DataLoader
    step: LrFinderStep
    config: LrFinderConfig


def run_range_test(
    inputs: RangeTestInput,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    spec = inputs.config.spec
    validate_lr_finder_spec(spec)
    device = resolve_device(inputs.config.runtime)
    amp = AmpSession.from_precision(inputs.config.runtime.precision, device)
    model = inputs.model
    optimizer = inputs.optimizer
    model.to(device)
    model.train()
    snapshot = snapshot_trainable(model, optimizer)
    try:
        lrs, losses, n_kept = sweep_learning_rates(
            model=model,
            optimizer=optimizer,
            train_loader=inputs.train_loader,
            step=inputs.step,
            spec=spec,
            device=device,
            amp=amp,
            enable_progress_bar=inputs.config.runtime.enable_progress_bar,
        )
    finally:
        restore_trainable(model, optimizer, snapshot, device)
    if n_kept < 2:
        raise LrFinderError("range test kept fewer than 2 points before diverging")
    return _kept(lrs, losses, n_kept)


def sweep_learning_rates(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    step: LrFinderStep,
    spec: LrFinderSpec,
    device: torch.device,
    amp: AmpSession,
    enable_progress_bar: bool,
) -> tuple[list[float], list[float], int]:
    num_iter = spec.num_iter
    lrs: list[float] = [0.0] * num_iter
    losses: list[float] = [0.0] * num_iter
    n_kept = 0
    avg_loss = 0.0
    best_loss = float("inf")
    gamma = spec.end_lr / spec.start_lr
    denom = float(num_iter - 1)
    batches = cycling_batches(train_loader, num_iter)
    steps = range_progress(
        num_iter,
        enabled=enable_progress_bar,
        desc="lr_finder",
        unit="step",
    )
    try:
        for i in steps:
            try:
                raw_batch = next(batches)
            except StopIteration as exc:
                raise LrFinderError("train loader is empty") from exc
            lr = spec.start_lr * (gamma ** (float(i) / denom))
            set_optimizer_lr(optimizer, lr)
            prepared = prepare_on_device(step, raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                output = step.forward_fn(model, prepared)
                loss = step.loss_fn(output, prepared)
            raw = float(loss.detach().cpu())
            if not math.isfinite(raw):
                break
            amp.optimizer_step(loss, optimizer)
            smoothed = _smooth_loss(
                raw, avg_loss=avg_loss, step_index=i, smooth_f=spec.smooth_f
            )
            if spec.smooth_f > 0.0:
                avg_loss = spec.smooth_f * avg_loss + (1.0 - spec.smooth_f) * raw
            if smoothed < best_loss:
                best_loss = smoothed
            if i > 0 and smoothed > spec.diverge_th * best_loss:
                break
            lrs[n_kept] = lr
            losses[n_kept] = smoothed
            n_kept += 1
            steps.set_postfix(lr=f"{lr:.2e}", loss=f"{smoothed:.4f}")
    finally:
        steps.close()
    return lrs, losses, n_kept


def _smooth_loss(
    raw: float, *, avg_loss: float, step_index: int, smooth_f: float
) -> float:
    if smooth_f <= 0.0:
        return raw
    updated = smooth_f * avg_loss + (1.0 - smooth_f) * raw
    return updated / (1.0 - smooth_f ** (step_index + 1))


def _kept(
    lrs: list[float], losses: list[float], n_kept: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    kept_lrs: list[float] = [0.0] * n_kept
    kept_losses: list[float] = [0.0] * n_kept
    for i in range(n_kept):
        kept_lrs[i] = lrs[i]
        kept_losses[i] = losses[i]
    return tuple(kept_lrs), tuple(kept_losses)
