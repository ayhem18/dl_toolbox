"""Physical GPU pool as nvidia-smi and CUDA_VISIBLE_DEVICES see it.

query_gpu_free_mib is the nvidia-smi table. bind_visible_gpus picks the N
freest devices for a Lightning trainer. query_free_gpu_indices lists every
device above a free-memory floor for wrapper child processes.
"""

from __future__ import annotations

import os
import subprocess

from dl_toolbox.base.parsing.parse_numerical import parse_positive_int


class VisibleGpus:
    """Env and unit names for the physical GPU pool."""

    CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
    MIB_PER_GIB = 1024


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
        raise ValueError(
            "nvidia-smi is not available; cannot select GPU devices"
        ) from exc
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


def candidate_indices(*, env_value: str | None) -> tuple[int, ...] | None:
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
                f"{VisibleGpus.CUDA_VISIBLE_DEVICES} has an empty index "
                f"in {env_value!r}"
            )
        try:
            index = int(token)
        except ValueError as exc:
            raise ValueError(
                f"{VisibleGpus.CUDA_VISIBLE_DEVICES} entries must be integer "
                f"GPU indices, got {token!r} in {env_value!r}"
            ) from exc
        if index in seen:
            raise ValueError(
                f"{VisibleGpus.CUDA_VISIBLE_DEVICES} has duplicate index "
                f"{index} in {env_value!r}"
            )
        seen[index] = None
        indices.append(index)
    return tuple(indices)


def select_freest_gpus(
    *,
    count: int,
    gpus: tuple[tuple[int, int], ...],
    candidates: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"need a GPU count >= 1, got {count!r}")
    pool = _pool_from_candidates(gpus=gpus, candidates=candidates)
    ordered = sorted(pool, key=lambda item: (-item[1], item[0]))
    n_pool = len(ordered)
    if n_pool < count:
        raise ValueError(f"need {count} GPU devices, found {n_pool}")
    selected: list[int] = [0] * count
    for i in range(count):
        selected[i] = ordered[i][0]
    return tuple(selected)


def bind_visible_gpus(*, count: int) -> tuple[int, ...]:
    selected = select_freest_gpus(
        count=count,
        gpus=query_gpu_free_mib(),
        candidates=candidate_indices(
            env_value=os.environ.get(VisibleGpus.CUDA_VISIBLE_DEVICES)
        ),
    )
    text = ",".join(str(index) for index in selected)
    os.environ[VisibleGpus.CUDA_VISIBLE_DEVICES] = text
    print(f"{VisibleGpus.CUDA_VISIBLE_DEVICES}={text}", flush=True)
    return selected


def gib_to_mib(*, min_free_gib: int) -> int:
    """Convert a GiB free-memory floor to the MiB units nvidia-smi reports."""
    return (
        parse_positive_int(min_free_gib, label="min_free_gib")
        * VisibleGpus.MIB_PER_GIB
    )


def free_gpu_indices(
    *,
    gpus: tuple[tuple[int, int], ...],
    candidates: tuple[int, ...] | None,
    min_free_mib: int,
) -> tuple[int, ...]:
    floor = parse_positive_int(min_free_mib, label="min_free_mib")
    pool = _pool_from_candidates(gpus=gpus, candidates=candidates)
    eligible: list[tuple[int, int]] = []
    n_pool = len(pool)
    for i in range(n_pool):
        index, free_mib = pool[i]
        if free_mib >= floor:
            eligible.append((index, free_mib))
    eligible.sort(key=lambda item: (-item[1], item[0]))
    n = len(eligible)
    indices: list[int] = [0] * n
    for i in range(n):
        indices[i] = eligible[i][0]
    return tuple(indices)


def query_free_gpu_indices(*, min_free_mib: int) -> tuple[int, ...]:
    return free_gpu_indices(
        gpus=query_gpu_free_mib(),
        candidates=candidate_indices(
            env_value=os.environ.get(VisibleGpus.CUDA_VISIBLE_DEVICES)
        ),
        min_free_mib=min_free_mib,
    )


def _pool_from_candidates(
    *,
    gpus: tuple[tuple[int, int], ...],
    candidates: tuple[int, ...] | None,
) -> tuple[tuple[int, int], ...]:
    if candidates is None:
        return gpus
    reported = {index: free for index, free in gpus}
    n_candidates = len(candidates)
    filtered: list[tuple[int, int]] = []
    for i in range(n_candidates):
        index = candidates[i]
        if index not in reported:
            raise ValueError(
                f"{VisibleGpus.CUDA_VISIBLE_DEVICES} index {index} is not a GPU "
                f"reported by nvidia-smi"
            )
        filtered.append((index, reported[index]))
    return tuple(filtered)
