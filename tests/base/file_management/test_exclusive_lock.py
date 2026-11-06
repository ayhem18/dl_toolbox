# PARKED: no contract yet. Not collected by pytest until this package is rewritten.
from __future__ import annotations

import pytest

pytest.skip("Parked: this package has no contract yet.", allow_module_level=True)

from pathlib import Path

import pytest

from dl_toolbox.base.file_management.exclusive_lock import (
    ExclusiveDirLock,
    ExclusiveLockTimeoutError,
)


def test_exclusive_dir_lock_acquire_release(tmp_path: Path) -> None:
    path = tmp_path / "catalog.lock"
    lock = ExclusiveDirLock(path, timeout_seconds=1.0)
    lock.acquire()
    assert path.is_dir()
    lock.release()
    assert not path.exists()


def test_exclusive_dir_lock_timeout(tmp_path: Path) -> None:
    path = tmp_path / "catalog.lock"
    held = ExclusiveDirLock(path, timeout_seconds=1.0)
    held.acquire()
    waiter = ExclusiveDirLock(path, timeout_seconds=0.2, poll_seconds=0.05)
    with pytest.raises(ExclusiveLockTimeoutError, match="timed out"):
        waiter.acquire()
    held.release()
    waiter.acquire()
    waiter.release()


def test_exclusive_dir_lock_rejects_relative_path() -> None:
    with pytest.raises(ValueError, match="absolute"):
        ExclusiveDirLock("catalog.lock", timeout_seconds=1.0)


def test_exclusive_dir_lock_context_manager(tmp_path: Path) -> None:
    path = tmp_path / "catalog.lock"
    with ExclusiveDirLock(path, timeout_seconds=1.0):
        assert path.is_dir()
    assert not path.exists()
