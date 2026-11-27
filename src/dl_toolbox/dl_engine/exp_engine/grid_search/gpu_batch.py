"""Run one train process per free GPU. The parent does not train."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from dl_toolbox.dl_engine.dl_engine_utils.visible_gpus import VisibleGpus

PopenFn = Callable[..., subprocess.Popen]
QueryFreeFn = Callable[[], tuple[int, ...]]


class GpuBatch:
    """Child-log conventions for run_recipes_on_free_gpus."""

    LOG_TAIL_LINES = 40


def run_recipes_on_free_gpus(
    *,
    recipes: Sequence[Path],
    train_script: Path,
    force: bool,
    tracking_backend: str | None,
    python: str,
    query_free: QueryFreeFn,
    popen: PopenFn,
) -> None:
    """Batch recipes onto currently free GPUs, wait, then re-query.

    Each child is ``train_script`` with ``CUDA_VISIBLE_DEVICES`` set to one
    physical index. The parent never calls ``engine.train``. GPUs already
    claimed in the current batch are not reused even if nvidia-smi has not
    yet shown the allocation. A child that exits non-zero does not skip
    later recipes; failures are raised after every recipe has been launched.
    """
    n_recipes = len(recipes)
    if n_recipes == 0:
        return
    failures: list[str] = []
    offset = 0
    while offset < n_recipes:
        free = query_free()
        n_free = len(free)
        if n_free == 0:
            raise RuntimeError(
                "no GPU with enough free memory to start a grid recipe"
            )
        remaining = n_recipes - offset
        batch_size = n_free if n_free < remaining else remaining
        procs: list[tuple[Path, Path, TextIO, subprocess.Popen]] = []
        claimed: dict[int, None] = {}
        for i in range(batch_size):
            gpu = _next_unclaimed(free, claimed)
            recipe = recipes[offset + i]
            procs.append(
                _launch_train(
                    python=python,
                    train_script=train_script,
                    recipe=recipe,
                    gpu_index=gpu,
                    force=force,
                    tracking_backend=tracking_backend,
                    popen=popen,
                )
            )
            claimed[gpu] = None
        failures.extend(_wait_batch(procs))
        offset += batch_size
    if len(failures) > 0:
        listed = "; ".join(failures)
        raise RuntimeError(f"grid train process failed: {listed}")


def _next_unclaimed(free: tuple[int, ...], claimed: dict[int, None]) -> int:
    n = len(free)
    for i in range(n):
        gpu = free[i]
        if gpu not in claimed:
            return gpu
    raise RuntimeError("batch is larger than the free GPU list")


def _launch_train(
    *,
    python: str,
    train_script: Path,
    recipe: Path,
    gpu_index: int,
    force: bool,
    tracking_backend: str | None,
    popen: PopenFn,
) -> tuple[Path, Path, TextIO, subprocess.Popen]:
    env = os.environ.copy()
    env[VisibleGpus.CUDA_VISIBLE_DEVICES] = str(gpu_index)
    env["PYTHONUNBUFFERED"] = "1"
    cmd: list[str] = [python, str(train_script), str(recipe)]
    if force:
        cmd.append("--force")
    if tracking_backend is not None:
        cmd.append("--tracking-backend")
        cmd.append(tracking_backend)
    log_path = recipe.with_suffix(".log")
    log_file = log_path.open("w", encoding="utf-8")
    log_file.write(f"gpu={gpu_index} cmd={' '.join(cmd)}\n")
    log_file.flush()
    print(f"grid child gpu={gpu_index} recipe={recipe.name} log={log_path}", flush=True)
    try:
        proc = popen(
            cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT
        )
    except BaseException:
        log_file.close()
        raise
    return recipe, log_path, log_file, proc


def _wait_batch(
    procs: list[tuple[Path, Path, TextIO, subprocess.Popen]],
) -> list[str]:
    failures: list[str] = []
    n = len(procs)
    for i in range(n):
        recipe, log_path, log_file, proc = procs[i]
        try:
            code = proc.wait()
        finally:
            log_file.close()
        if code != 0:
            tail = _log_tail(log_path, n_lines=GpuBatch.LOG_TAIL_LINES)
            failures.append(
                f"{recipe} (exit {code}; log {log_path})\n{tail}".rstrip()
            )
    return failures


def _log_tail(path: Path, *, n_lines: int) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    n = len(lines)
    start = 0 if n <= n_lines else n - n_lines
    return "\n".join(lines[start:n])
