"""Pick the N GPUs with the most free memory, then bind CUDA_VISIBLE_DEVICES."""

from __future__ import annotations

import os
import subprocess

from dl_toolbox.dl_engine.exp_engine.process_rank import ProcessRank

CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"


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


def candidate_indices(env_value: str | None) -> tuple[int, ...] | None:
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
) -> tuple[int, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"need a GPU count >= 1, got {count!r}")
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
        candidates=candidate_indices(os.environ.get(CUDA_VISIBLE_DEVICES)),
    )
    text = ",".join(str(index) for index in selected)
    os.environ[CUDA_VISIBLE_DEVICES] = text
    print(f"{CUDA_VISIBLE_DEVICES}={text}", flush=True)
    return selected


def distributed_launch_started() -> bool:
    names = (
        ProcessRank.LOCAL_RANK_ENV,
        ProcessRank.RANK_ENV,
        ProcessRank.GLOBAL_RANK_ENV,
    )
    n = len(names)
    for i in range(n):
        raw = os.environ.get(names[i])
        if raw is None:
            continue
        if raw.strip() == "":
            continue
        return True
    return False
