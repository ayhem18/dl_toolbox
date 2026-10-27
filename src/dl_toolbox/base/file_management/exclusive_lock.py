"""Process-exclusive lock via atomic mkdir. Works on Windows and POSIX."""

from __future__ import annotations

import time
from pathlib import Path


class ExclusiveLockTimeoutError(TimeoutError):
    """The lock directory could not be created before the timeout."""


class ExclusiveDirLock:
    """Exclusive lock: ``path.mkdir()`` succeeds for one process.

    ``path`` is a directory. Crash while holding the lock leaves that
    directory behind; the next acquire waits until ``timeout_seconds``.
    """

    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: float,
        poll_seconds: float = 0.05,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}")
        if poll_seconds <= 0:
            raise ValueError(f"poll_seconds must be > 0, got {poll_seconds}")
        self._path = path
        self._timeout_seconds = float(timeout_seconds)
        self._poll_seconds = float(poll_seconds)
        self._held = False

    def acquire(self) -> None:
        if self._held:
            raise RuntimeError(f"lock already held: {self._path}")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self._timeout_seconds
        while True:
            try:
                self._path.mkdir()
                self._held = True
                return
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise ExclusiveLockTimeoutError(
                        f"timed out waiting for lock {self._path}"
                    )
                time.sleep(self._poll_seconds)

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        try:
            self._path.rmdir()
        except FileNotFoundError:
            return

    def __enter__(self) -> ExclusiveDirLock:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        self.release()
        return False
