"""Lock, catalog, prune, snapshot. No Lightning. No inspect folders."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dl_toolbox.base.file_management.exclusive_lock import ExclusiveDirLock
from dl_toolbox.base.input_output.json_io import read_json, write_json
from dl_toolbox.dl_engine.training.checkpointer.hashing import sha256_file
from dl_toolbox.dl_engine.training.checkpointer.types import (
    CATALOG_FILENAME,
    CHECKPOINTS_DIRNAME,
    FIT_CLAIMED,
    FIT_FINISHED,
    FIT_RUNNING,
    KIND_EPOCH,
    KIND_LAST,
    KIND_RANK,
    KIND_VALUE,
    LIVE_DIRNAME,
    LOCK_DIRNAME,
    CheckpointLookupError,
    CheckpointPolicy,
    CheckpointRecord,
    CheckpointRequest,
    FitRunState,
)
from dl_toolbox.dl_engine.training.config import MonitorConfig

_ID_LEN = 12
_DEFAULT_LOCK_TIMEOUT = 600.0


class CheckpointManager:
    def __init__(self, *, lock_timeout_seconds: float = _DEFAULT_LOCK_TIMEOUT) -> None:
        timeout = float(lock_timeout_seconds)
        if timeout <= 0:
            raise ValueError(f"lock_timeout_seconds must be > 0, got {lock_timeout_seconds}")
        self._lock_timeout_seconds = timeout

    @property
    def lock_timeout_seconds(self) -> float:
        return self._lock_timeout_seconds

    def bind(
        self,
        *,
        run_dir: Path,
        policy: CheckpointPolicy,
        monitor: MonitorConfig,
        max_epochs: int,
    ) -> "CheckpointerBinding":
        return CheckpointerBinding(
            manager=self,
            run_dir=run_dir,
            policy=policy,
            monitor=monitor,
            max_epochs=max_epochs,
            lock_timeout_seconds=self._lock_timeout_seconds,
        )


class CheckpointerBinding:
    """Run-bound filesystem algorithm. Checkpointer is the façade."""

    def __init__(
        self,
        *,
        manager: CheckpointManager,
        run_dir: Path,
        policy: CheckpointPolicy,
        monitor: MonitorConfig,
        max_epochs: int,
        lock_timeout_seconds: float,
    ) -> None:
        del manager
        if policy.save_top_k < -1:
            raise ValueError(f"save_top_k must be >= -1, got {policy.save_top_k}")
        if monitor.mode not in ("min", "max"):
            raise ValueError(f"monitor.mode must be 'min' or 'max', got {monitor.mode!r}")
        if not run_dir.is_absolute():
            raise ValueError(f"run_dir must be an absolute path, got {run_dir}")
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {max_epochs}")
        self._run_dir = run_dir
        self._policy = policy
        self._monitor = monitor
        self._max_epochs = max_epochs
        self._root = run_dir / CHECKPOINTS_DIRNAME
        self._live = self._root / LIVE_DIRNAME
        self._catalog_path = self._root / CATALOG_FILENAME
        self._lock = ExclusiveDirLock(
            self._root / LOCK_DIRNAME,
            timeout_seconds=lock_timeout_seconds,
        )

    @property
    def monitor_key(self) -> str:
        return self._monitor.key

    def save_new(self, *, epoch: int, score: float, write: Callable[[Path], None]) -> None:
        self._live.mkdir(parents=True, exist_ok=True)
        checkpoint_id = _new_id()
        path = self._live / f"{checkpoint_id}.ckpt"
        committed = False
        self._lock.acquire()
        try:
            write(path)
            if not path.is_file():
                raise FileNotFoundError(f"save_checkpoint did not write {path}")
            digest = sha256_file(path)
            size = path.stat().st_size
            catalog = self._read_or_empty()
            self._commit_new(
                catalog,
                epoch=epoch,
                score=score,
                checkpoint_id=checkpoint_id,
                rel_path=f"{LIVE_DIRNAME}/{checkpoint_id}.ckpt",
                digest=digest,
                size=size,
            )
            write_json(self._catalog_path, catalog)
            committed = True
        finally:
            if not committed and path.exists():
                path.unlink()
            self._lock.release()

    def fetch(self, request: CheckpointRequest, *, dest: Path) -> CheckpointRecord:
        dest.mkdir(parents=True, exist_ok=True)
        self._lock.acquire()
        copied: Path | None = None
        try:
            catalog = self._read_catalog()
            row = self._resolve(catalog, request)
            source = self._root / str(row["path"])
            if not source.is_file():
                raise CheckpointLookupError(f"catalog entry is missing on disk: {source}")
            copied = dest / source.name
            shutil.copy2(source, copied)
            digest = sha256_file(copied)
            size = copied.stat().st_size
            if digest != row["sha256"] or size != int(row["bytes"]):
                copied.unlink()
                copied = None
                raise CheckpointLookupError(
                    f"checksum mismatch for epoch {row['epoch']} at {source}"
                )
            return CheckpointRecord(
                epoch=int(row["epoch"]),
                path=copied,
                rank=row.get("rank"),
                score=float(row["score"]),
                kind=request.kind,
                checkpoint_id=str(row["id"]),
            )
        finally:
            self._lock.release()

    def fit_state(self) -> FitRunState:
        if not self._catalog_path.is_file():
            return FitRunState(
                status=FIT_CLAIMED,
                latest_epoch=None,
                current_best_epoch=None,
                max_epochs=self._max_epochs,
            )
        catalog = read_json(self._catalog_path)
        fit = catalog.get("fit") or {}
        best = _best_epoch(catalog)
        latest = fit.get("latest_epoch")
        return FitRunState(
            status=str(fit.get("status") or FIT_RUNNING),
            latest_epoch=None if latest is None else int(latest),
            current_best_epoch=best,
            max_epochs=int(fit.get("max_epochs", self._max_epochs)),
        )

    def mark_finished(self) -> None:
        self._lock.acquire()
        try:
            catalog = self._read_or_empty()
            catalog["fit"]["status"] = FIT_FINISHED
            write_json(self._catalog_path, catalog)
        finally:
            self._lock.release()

    def _read_catalog(self) -> dict[str, Any]:
        if not self._catalog_path.is_file():
            raise CheckpointLookupError(f"catalog does not exist: {self._catalog_path}")
        return read_json(self._catalog_path)

    def _read_or_empty(self) -> dict[str, Any]:
        if self._catalog_path.is_file():
            return read_json(self._catalog_path)
        return self._empty_catalog()

    def _empty_catalog(self) -> dict[str, Any]:
        return {
            "monitor": {"key": self._monitor.key, "mode": self._monitor.mode},
            "policy": {
                "save_top_k": self._policy.save_top_k,
                "save_last": self._policy.save_last,
            },
            "fit": {
                "status": FIT_RUNNING,
                "latest_epoch": None,
                "max_epochs": self._max_epochs,
            },
            "current": {"files": [], "last": None},
            "history": {},
        }

    def _commit_new(
        self,
        catalog: dict[str, Any],
        *,
        epoch: int,
        score: float,
        checkpoint_id: str,
        rel_path: str,
        digest: str,
        size: int,
    ) -> None:
        current_files = list(catalog.get("current", {}).get("files") or [])
        history = dict(catalog.get("history") or {})
        candidates = [dict(row) for row in current_files]
        new_row = {
            "epoch": epoch,
            "id": checkpoint_id,
            "path": rel_path,
            "score": score,
            "sha256": digest,
            "bytes": size,
        }
        candidates.append(new_row)
        reverse = self._monitor.mode == "max"
        candidates.sort(key=lambda item: float(item["score"]), reverse=reverse)
        n = len(candidates)
        for i in range(n):
            candidates[i]["rank"] = i + 1
        keep_n = n if self._policy.save_top_k == -1 else self._policy.save_top_k
        ranked = candidates[:keep_n] if keep_n >= 0 else []
        keep_ids = {str(row["id"]) for row in ranked}
        last_row = None
        if self._policy.save_last:
            last_row = {"epoch": epoch, "id": checkpoint_id}
            keep_ids.add(checkpoint_id)
        live_files = list(self._live.glob("*.ckpt")) if self._live.is_dir() else []
        n_live = len(live_files)
        for i in range(n_live):
            live_path = live_files[i]
            live_id = live_path.stem
            if live_id not in keep_ids:
                live_path.unlink()
        history_keys = list(history)
        n_hist = len(history_keys)
        for i in range(n_hist):
            key = history_keys[i]
            entry = dict(history[key])
            entry["retained"] = str(entry.get("id")) in keep_ids
            history[key] = entry
        new_rank = None
        n_ranked = len(ranked)
        for i in range(n_ranked):
            if str(ranked[i]["id"]) == checkpoint_id:
                new_rank = int(ranked[i]["rank"])
                break
        history[str(epoch)] = {
            "id": checkpoint_id,
            "score": score,
            "rank_then": new_rank if new_rank is not None else int(new_row.get("rank") or n),
            "sha256": digest,
            "retained": checkpoint_id in keep_ids,
        }
        catalog["current"] = {"files": ranked, "last": last_row}
        catalog["history"] = history
        catalog["fit"] = {
            "status": FIT_RUNNING,
            "latest_epoch": epoch,
            "max_epochs": self._max_epochs,
        }
        catalog["monitor"] = {"key": self._monitor.key, "mode": self._monitor.mode}
        catalog["policy"] = {
            "save_top_k": self._policy.save_top_k,
            "save_last": self._policy.save_last,
        }

    def _resolve(
        self,
        catalog: dict[str, Any],
        request: CheckpointRequest,
    ) -> dict[str, Any]:
        files = list(catalog.get("current", {}).get("files") or [])
        last = catalog.get("current", {}).get("last")
        if request.kind == KIND_LAST:
            if last is None:
                raise CheckpointLookupError("selector 'last' is not in the catalog")
            return self._row_by_id(files, last, catalog)
        if request.kind == KIND_RANK:
            rank = request.rank
            if rank is None:
                raise CheckpointLookupError("rank request is missing rank")
            matched = [row for row in files if row.get("rank") == rank]
            if not matched:
                raise CheckpointLookupError(f"selector 'rank={rank}' is not in the catalog")
            return matched[0]
        if request.kind == KIND_EPOCH:
            epoch = request.epoch
            if epoch is None:
                raise CheckpointLookupError("epoch request is missing epoch")
            matched = [row for row in files if int(row["epoch"]) == epoch]
            if not matched:
                raise CheckpointLookupError(f"selector 'epoch={epoch}' is not in the catalog")
            return matched[0]
        if request.kind == KIND_VALUE:
            value = request.value
            if value is None:
                raise CheckpointLookupError("value request is missing value")
            matched = [row for row in files if float(row["score"]) == value]
            if not matched:
                raise CheckpointLookupError(f"selector 'value={value}' is not in the catalog")
            return matched[0]
        raise CheckpointLookupError(f"unknown request kind {request.kind!r}")

    def _row_by_id(
        self,
        files: list[dict[str, Any]],
        last: dict[str, Any],
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        last_id = str(last["id"])
        n = len(files)
        for i in range(n):
            if str(files[i]["id"]) == last_id:
                return files[i]
        history = catalog.get("history") or {}
        epoch = str(last["epoch"])
        entry = history.get(epoch)
        if entry is None or str(entry.get("id")) != last_id:
            raise CheckpointLookupError("selector 'last' is not in the catalog")
        return {
            "epoch": int(last["epoch"]),
            "id": last_id,
            "path": f"{LIVE_DIRNAME}/{last_id}.ckpt",
            "score": float(entry["score"]),
            "rank": None,
            "sha256": entry["sha256"],
            "bytes": (self._live / f"{last_id}.ckpt").stat().st_size
            if (self._live / f"{last_id}.ckpt").is_file()
            else -1,
        }


def _new_id() -> str:
    return uuid.uuid4().hex[:_ID_LEN]


def _best_epoch(catalog: dict[str, Any]) -> int | None:
    files = list(catalog.get("current", {}).get("files") or [])
    n = len(files)
    for i in range(n):
        if files[i].get("rank") == 1:
            return int(files[i]["epoch"])
    return None
