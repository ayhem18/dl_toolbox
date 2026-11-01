"""Interpret trainer ``devices`` as a count of GPUs and bind the freest ones.

``devices`` in the recipe stays an integer. This module queries nvidia-smi,
keeps GPUs with at least 4 GiB free, takes the top-N, and writes
``CUDA_VISIBLE_DEVICES`` so Lightning sees that many devices.
"""

from __future__ import annotations

import os
import subprocess

CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
MIN_FREE_GIB = 4
MIN_FREE_MIB = MIN_FREE_GIB * 1024


def query_gpu_free_mib() -> tuple[tuple[int, int], ...]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,nounits,noheader",
            ],
            encoding="utf-8",
        )
    except FileNotFoundError as exc:
        raise ValueError("nvidia-smi is not available; cannot select GPU devices") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"nvidia-smi failed: {exc}") from exc
    lines = output.strip().splitlines()
    n_lines = len(lines)
    gpus: list[tuple[int, int]] = []
    for i in range(n_lines):
        line = lines[i].strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 2:
            raise ValueError(f"unexpected nvidia-smi row {line!r}")
        gpus.append((int(parts[0].strip()), int(parts[1].strip())))
    if not gpus:
        raise ValueError("nvidia-smi reported no GPUs")
    return tuple(gpus)


def candidate_indices(env_value: str | None) -> tuple[int, ...] | None:
    """Physical GPU indices already listed in ``CUDA_VISIBLE_DEVICES``.

    ``None`` means the variable is unset: every GPU nvidia-smi reports is a
    candidate.
    """
    if env_value is None:
        return None
    text = env_value.strip()
    if not text:
        return None
    parts = text.split(",")
    n_parts = len(parts)
    indices: list[int] = []
    seen: dict[int, None] = {}
    for i in range(n_parts):
        token = parts[i].strip()
        if not token:
            raise ValueError(
                f"{CUDA_VISIBLE_DEVICES} has an empty index in {env_value!r}"
            )
        try:
            index = int(token)
        except ValueError as exc:
            raise ValueError(
                f"{CUDA_VISIBLE_DEVICES} entries must be integer GPU indices, "
                f"got {token!r} in {env_value!r}"
            ) from exc
        if index in seen:
            raise ValueError(
                f"{CUDA_VISIBLE_DEVICES} has duplicate index {index} in {env_value!r}"
            )
        seen[index] = None
        indices.append(index)
    return tuple(indices)


def select_freest_gpus(
    *,
    count: int,
    gpus: tuple[tuple[int, int], ...],
    candidates: tuple[int, ...] | None,
    min_free_mib: int = MIN_FREE_MIB,
) -> tuple[int, ...]:
    if count < 1:
        raise ValueError(f"devices must be >= 1, got {count}")
    if min_free_mib < 1:
        raise ValueError(f"min_free_mib must be >= 1, got {min_free_mib}")
    pool = gpus
    if candidates is not None:
        reported = {index: free for index, free in gpus}
        n_candidates = len(candidates)
        filtered: list[tuple[int, int]] = []
        for i in range(n_candidates):
            index = candidates[i]
            if index not in reported:
                raise ValueError(
                    f"{CUDA_VISIBLE_DEVICES} index {index} is not a GPU "
                    f"reported by nvidia-smi"
                )
            filtered.append((index, reported[index]))
        pool = tuple(filtered)
    eligible: list[tuple[int, int]] = []
    n_pool = len(pool)
    for i in range(n_pool):
        index, free_mib = pool[i]
        if free_mib >= min_free_mib:
            eligible.append((index, free_mib))
    eligible.sort(key=lambda item: (-item[1], item[0]))
    if len(eligible) < count:
        detail = ", ".join(
            f"GPU {index}: {free_mib} MiB free" for index, free_mib in pool
        )
        raise ValueError(
            f"need {count} GPU(s) with at least {min_free_mib} MiB free; "
            f"eligible={len(eligible)} ({detail})"
        )
    selected = [eligible[i][0] for i in range(count)]
    return tuple(selected)


def bind_visible_gpus(*, count: int) -> tuple[int, ...]:
    selected = select_freest_gpus(
        count=count,
        gpus=query_gpu_free_mib(),
        candidates=candidate_indices(os.environ.get(CUDA_VISIBLE_DEVICES)),
    )
    os.environ[CUDA_VISIBLE_DEVICES] = ",".join(str(index) for index in selected)
    return selected
