"""ProcessRank is this process's distributed rank as Lightning / torch write it.

is_ddp_worker is true when any named rank variable parses as an int > 0.
A missing name, an empty string, or a non-integer string is unset.
Unset is not a worker.
distributed_launch_started is true when LOCAL_RANK, RANK, or GLOBAL_RANK is set
to a non-empty string. Rank 0 after torchrun is a started launch and not a worker.
"""

from __future__ import annotations

import os


class ProcessRank:
    """Rank environment names and is_ddp_worker."""

    RANK_ENV = "RANK"
    GLOBAL_RANK_ENV = "GLOBAL_RANK"
    NODE_RANK_ENV = "NODE_RANK"
    GROUP_RANK_ENV = "GROUP_RANK"
    LOCAL_RANK_ENV = "LOCAL_RANK"

    @staticmethod
    def is_ddp_worker() -> bool:
        rank = _env_int(ProcessRank.RANK_ENV)
        if rank is None:
            rank = _env_int(ProcessRank.GLOBAL_RANK_ENV)
        if rank is not None and rank > 0:
            return True
        node = _env_int(ProcessRank.NODE_RANK_ENV)
        if node is None:
            node = _env_int(ProcessRank.GROUP_RANK_ENV)
        if node is not None and node > 0:
            return True
        local = _env_int(ProcessRank.LOCAL_RANK_ENV)
        if local is not None and local > 0:
            return True
        return False

    @staticmethod
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


def is_ddp_worker() -> bool:
    return ProcessRank.is_ddp_worker()


def distributed_launch_started() -> bool:
    return ProcessRank.distributed_launch_started()


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    text = raw.strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError:
        return None


__all__ = [
    "ProcessRank",
    "distributed_launch_started",
    "is_ddp_worker",
]
