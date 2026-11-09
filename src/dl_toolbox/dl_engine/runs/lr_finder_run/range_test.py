"""RANGE and FULL_DS sweeps, then suggested LRs from the recorded curve."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from dl_toolbox.dl_engine.dl_engine_utils.batches import cycling_batches, mapping_to_device
from dl_toolbox.dl_engine.dl_engine_utils.optimizer_lr import set_optimizer_lr
from dl_toolbox.dl_engine.dl_engine_utils.precision.amp import AmpSession
from dl_toolbox.dl_engine.dl_engine_utils.trainable_snapshot import (
    TrainableSnapshot,
    restore_trainable,
)
from dl_toolbox.dl_engine.runs.lr_finder_run.spec import (
    FindLrError,
    FindLrRuntime,
    FindLrSpec,
    FindLrSteps,
    log_spaced_lrs,
)


def next_cycled_batch(batches: Iterator[Any]) -> Any:
    try:
        return next(batches)
    except ValueError as exc:
        raise FindLrError("train loader is empty") from exc


def prepare_batch(
    *, steps: FindLrSteps, raw_batch: Any, device: torch.device
) -> dict[str, Any]:
    prepared = steps.prepare_tensors_fn(raw_batch)
    if not isinstance(prepared, dict):
        raise TypeError(
            f"prepare_tensors_fn must return a dict, got {type(prepared)!r}"
        )
    return mapping_to_device(mapping=prepared, device=device)


def run_range_test(
    *,
    net: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    steps: FindLrSteps,
    spec: FindLrSpec,
    device: torch.device,
    runtime: FindLrRuntime,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if spec.kind != FindLrSpec.RANGE:
        raise FindLrError(
            f"run_range_test requires kind {FindLrSpec.RANGE!r}, got {spec.kind!r}"
        )
    net.train()
    amp = AmpSession.from_precision(runtime.precision, device)
    candidates = log_spaced_lrs(spec)
    batches = cycling_batches(loader=train_loader, num_iter=spec.num_iter)
    avg_loss = 0.0
    best_loss = math.inf
    kept_lrs: list[float] = []
    kept_losses: list[float] = []
    progress = tqdm(
        range(spec.num_iter),
        desc="lr_finder",
        unit="step",
        disable=not runtime.enable_progress_bar,
    )
    try:
        for i in progress:
            raw_batch = next_cycled_batch(batches)
            lr = candidates[i]
            set_optimizer_lr(optimizer=optimizer, lr=lr)
            prepared = prepare_batch(steps=steps, raw_batch=raw_batch, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = _forward_loss(net=net, steps=steps, prepared=prepared, amp=amp)
            if loss is None:
                break
            raw = float(loss.detach().cpu())
            if not math.isfinite(raw):
                break
            amp.optimizer_step(loss, optimizer)
            if spec.smooth_f <= 0.0:
                smoothed = raw
            else:
                avg_loss = spec.smooth_f * avg_loss + (1.0 - spec.smooth_f) * raw
                smoothed = avg_loss / (1.0 - spec.smooth_f ** (i + 1))
            if smoothed < best_loss:
                best_loss = smoothed
            if i > 0 and smoothed > spec.diverge_th * best_loss:
                break
            kept_lrs.append(lr)
            kept_losses.append(smoothed)
    finally:
        progress.close()
    if len(kept_lrs) < 2:
        raise FindLrError("range test kept fewer than 2 points before diverging")
    return tuple(kept_lrs), tuple(kept_losses)


def run_full_ds_range_test(
    *,
    net: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    steps: FindLrSteps,
    spec: FindLrSpec,
    device: torch.device,
    runtime: FindLrRuntime,
    snapshot: TrainableSnapshot,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if spec.kind != FindLrSpec.FULL_DS:
        raise FindLrError(
            f"run_full_ds_range_test requires kind {FindLrSpec.FULL_DS!r}, "
            f"got {spec.kind!r}"
        )
    candidates = log_spaced_lrs(spec)
    amp = AmpSession.from_precision(runtime.precision, device)
    kept_lrs: list[float] = []
    kept_losses: list[float] = []
    progress = tqdm(
        range(spec.num_iter),
        desc="full_ds_lr_finder",
        unit="lr",
        disable=not runtime.enable_progress_bar,
    )
    try:
        for i in progress:
            lr = candidates[i]
            restore_trainable(
                model=net, optimizer=optimizer, snapshot=snapshot, device=device
            )
            set_optimizer_lr(optimizer=optimizer, lr=lr)
            train_ok = True
            for _pass in range(spec.num_iterations_per_lr):
                train_ok = _train_one_pass(
                    net=net,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    steps=steps,
                    amp=amp,
                    device=device,
                )
                if not train_ok:
                    break
            if not train_ok:
                continue
            eval_loss = _eval_mean_loss(
                net=net,
                train_loader=train_loader,
                steps=steps,
                amp=amp,
                device=device,
            )
            if eval_loss is None:
                continue
            kept_lrs.append(lr)
            kept_losses.append(eval_loss)
    finally:
        progress.close()
    if len(kept_lrs) < 1:
        raise FindLrError("full-ds range test kept no finite eval losses")
    return tuple(kept_lrs), tuple(kept_losses)


def _train_one_pass(
    *,
    net: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    steps: FindLrSteps,
    amp: AmpSession,
    device: torch.device,
) -> bool:
    net.train()
    n_batches = 0
    for raw_batch in train_loader:
        prepared = prepare_batch(steps=steps, raw_batch=raw_batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = _forward_loss(net=net, steps=steps, prepared=prepared, amp=amp)
        if loss is None:
            return False
        raw = float(loss.detach().cpu())
        if not math.isfinite(raw):
            return False
        amp.optimizer_step(loss, optimizer)
        n_batches += 1
    if n_batches == 0:
        raise FindLrError("train loader is empty")
    return True


def _eval_mean_loss(
    *,
    net: nn.Module,
    train_loader: DataLoader,
    steps: FindLrSteps,
    amp: AmpSession,
    device: torch.device,
) -> float | None:
    net.eval()
    total = 0.0
    n_eval = 0
    with torch.no_grad():
        for raw_batch in train_loader:
            prepared = prepare_batch(steps=steps, raw_batch=raw_batch, device=device)
            loss = _forward_loss(net=net, steps=steps, prepared=prepared, amp=amp)
            if loss is None:
                return None
            raw = float(loss.detach().cpu())
            if not math.isfinite(raw):
                return None
            total += raw
            n_eval += 1
    if n_eval == 0:
        raise FindLrError("train loader is empty")
    return total / float(n_eval)


def _forward_loss(
    *,
    net: nn.Module,
    steps: FindLrSteps,
    prepared: dict,
    amp: AmpSession,
) -> Tensor | None:
    try:
        with amp.autocast():
            output = steps.forward_fn(net, prepared)
            if (
                isinstance(output, Tensor)
                and output.is_floating_point()
                and not bool(output.isfinite().all())
            ):
                return None
            return steps.loss_fn(output, prepared)
    except FindLrError:
        raise
    except ValueError:
        return None


def suggest_lrs(
    lrs: Sequence[float],
    losses: Sequence[float],
) -> dict[str, float]:
    _require_curve(lrs, losses, min_points=2)
    min_i = _first_min_index(losses)
    if min_i < 1:
        raise FindLrError(
            "loss minimum is the first recorded point; widen the LR range"
        )
    steepest_i = _steepest_index(lrs, losses, stop=min_i)
    min_lr = float(lrs[min_i])
    return {
        "steepest": float(lrs[steepest_i]),
        "before_min": min_lr / 10.0,
        "min_loss_lr": min_lr,
    }


def suggest_min_loss_lr(
    lrs: Sequence[float],
    losses: Sequence[float],
) -> dict[str, float]:
    _require_curve(lrs, losses, min_points=1)
    min_i = _first_min_index(losses)
    return {"min_loss_lr": float(lrs[min_i])}


def _require_curve(
    lrs: Sequence[float],
    losses: Sequence[float],
    *,
    min_points: int,
) -> None:
    n = len(lrs)
    if n != len(losses):
        raise FindLrError("lrs and losses must be the same length")
    if n < min_points:
        raise FindLrError(
            f"need at least {min_points} curve point(s) to suggest an LR"
        )
    for i in range(n):
        if not math.isfinite(float(losses[i])):
            raise FindLrError("curve losses must be finite")


def _first_min_index(losses: Sequence[float]) -> int:
    n = len(losses)
    best_i = 0
    best = float(losses[0])
    for i in range(1, n):
        value = float(losses[i])
        if value < best:
            best = value
            best_i = i
    return best_i


def _steepest_index(
    lrs: Sequence[float],
    losses: Sequence[float],
    *,
    stop: int,
) -> int:
    best_i = 0
    best_grad = math.inf
    found = False
    for i in range(stop):
        width = math.log10(float(lrs[i + 1])) - math.log10(float(lrs[i]))
        if width <= 0.0:
            continue
        grad = (float(losses[i + 1]) - float(losses[i])) / width
        if not found or grad < best_grad:
            best_grad = grad
            best_i = i
            found = True
    if not found:
        raise FindLrError("cannot compute d(loss)/d(log10 lr) before the minimum")
    return best_i
