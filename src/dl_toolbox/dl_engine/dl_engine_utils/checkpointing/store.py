"""Disk reflection of the Checkpoint Directory entity.

The entity is the directory ``{run_artifacts_dir}/checkpoints/``. This class
does not invent a second in-memory model: after every successful mutation the
files on disk *are* the entity. ``InspectableCheckpointer`` is the accessor;
it takes ``ExclusiveDirLock`` on ``catalog.lock`` before calling these methods.
This store does not take the lock.

Layout
------
::

    run_artifacts_dir/                 Directory, must already exist
      checkpoints/                     created on first locked write
        catalog.lock/                  ExclusiveDirLock (mkdir lock, not a File)
        catalog.json                   named state (CatalogDocument)
        live/
          {id}.ckpt                    retained payloads (File)
          .ddp_join.ckpt               Lightning DDP join path, not catalogued

What each file means
--------------------
- **Missing ``catalog.json``:** fit has been claimed but has not yet written
  a checkpoint (or a skip stamp). ``fit_state`` reports ``claimed``.
- **``catalog.json``:** the spec of the entity right now: monitor, policy,
  fit status / latest epoch, the ranked keep-set (``current.files``), the
  last pointer, and per-written-epoch history.
- **``live/{id}.ckpt``:** payload for an id that policy kept. Every id in
  ``current.files`` and ``current.last`` must be a File whose sha256 and
  size match the catalog row. The catalog is the index; the files are the
  bytes. A missing file or a digest mismatch is an error, not repaired.
- **``catalog.lock``:** process-exclusive access. Rank-0 of a DDP fit and
  the inspect process compete here. Other DDP ranks do not take this lock;
  they use Lightning's ``strategy.barrier`` inside ``Trainer.save_checkpoint``.

Reference flows (caller holds the lock)
---------------------------------------
Save::

    catalog = read_or_empty()
    if not decide_keep(catalog.current.files, score):
        stamp latest_epoch, write catalog, return  # no live file
    write live/{new_id}.ckpt via write_function
    hash File, apply_policy, prune live stems not in keep_ids
    write catalog

Fetch::

    parse selector against current.files (epoch / rank / value)
    or against current.last (file may sit outside top-k; history supplies
    score and sha256)
    copy File to dest Directory, verify hash_file + size

Mark finished::

    read_or_empty, set fit.status = finished, write catalog

Invariants you can check against this code
------------------------------------------
1. ``current.files`` is the ranked top-k only. A ``save_last`` file that is
   not in top-k is **not** in ``current.files``. The next ``decide_keep``
   therefore does not rank against it.
2. A skipped epoch updates ``fit.latest_epoch`` and does **not** add
   history or a live file. Fetch of that epoch fails.
3. Fetch by epoch / rank / value searches ``current.files`` only. History
   is not a fetch index except when resolving ``last`` outside top-k.
4. Prune deletes every ``live/*.ckpt`` whose stem is not in the keep set,
   including orphans that are not in the catalog. ``.ddp_join.ckpt`` has a
   stem that is never a keep id, so it is deleted on the next kept save.
5. Equal scores: sort is stable, the older row keeps the better rank. A new
   equal score with ``save_top_k=1`` is skipped.
6. ``run_artifacts_dir`` is a Directory that already exists. ``live/`` and
   the fetch destination are Directory CREATE. Catalog and ckpt paths that
   we read are File (existence is an invariant).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
    DirectorySpec,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.hashing.identity import hash_file
from dl_toolbox.base.input_output.json_io import JsonTextForm, read_object, write_object
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.policy import (
    RankedCheckpoint,
    ScoredCheckpoint,
    apply_policy,
)
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.types import (
    CATALOG_FILENAME,
    CHECKPOINTS_DIRNAME,
    DDP_JOIN_FILENAME,
    FIT_RUNNING,
    KIND_EPOCH,
    KIND_LAST,
    KIND_RANK,
    KIND_VALUE,
    LIVE_DIRNAME,
    LOCK_DIRNAME,
    CatalogCurrent,
    CatalogDocument,
    CatalogFit,
    CatalogHistoryEntry,
    CatalogLiveRow,
    CatalogLastPointer,
    CheckpointLookupError,
    CheckpointMonitor,
    CheckpointPolicy,
    CheckpointRecord,
    CheckpointRequest,
)


class CheckpointStore:
    def __init__(
        self,
        *,
        run_artifacts_dir: Directory,
        policy: CheckpointPolicy,
        monitor: CheckpointMonitor,
        max_epochs: int,
    ) -> None:
        if run_artifacts_dir.spec != DirectorySpec.EXISTS:
            raise ValueError(
                "run_artifacts_dir Directory must exist: "
                f"{run_artifacts_dir.path}"
            )
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {max_epochs}")
        self._run_artifacts_dir = run_artifacts_dir
        self._policy = policy
        self._monitor = monitor
        self._max_epochs = max_epochs
        self._root = run_artifacts_dir.path / CHECKPOINTS_DIRNAME
        self._live = self._root / LIVE_DIRNAME
        self._catalog_path = self._root / CATALOG_FILENAME

    @property
    def root(self) -> Path:
        return self._root

    @property
    def live(self) -> Path:
        return self._live

    @property
    def catalog_path(self) -> Path:
        return self._catalog_path

    @property
    def lock_dir(self) -> Path:
        return self._root / LOCK_DIRNAME

    def live_file_path(self, checkpoint_id: str) -> Path:
        return self._live / f"{checkpoint_id}.ckpt"

    def ddp_join_path(self) -> Path:
        self.ensure_live()
        return self._live / DDP_JOIN_FILENAME

    def ensure_root(self) -> Directory:
        return Directory.from_absolute(
            path=self._root, policy=DirectoryParsePolicy.CREATE
        )

    def ensure_live(self) -> Directory:
        self.ensure_root()
        return Directory.from_absolute(
            path=self._live, policy=DirectoryParsePolicy.CREATE
        )

    def catalog_exists(self) -> bool:
        return self._catalog_path.is_file()

    def read_catalog(self) -> CatalogDocument:
        if not self._catalog_path.is_file():
            raise CheckpointLookupError(
                f"catalog does not exist: {self._catalog_path}"
            )
        return CatalogDocument.model_validate(
            read_object(file=File.from_absolute(path=self._catalog_path))
        )

    def read_or_empty(self) -> CatalogDocument:
        if self._catalog_path.is_file():
            return CatalogDocument.model_validate(
                read_object(file=File.from_absolute(path=self._catalog_path))
            )
        return self.empty_catalog()

    def empty_catalog(self) -> CatalogDocument:
        return CatalogDocument(
            monitor=self._monitor,
            policy=self._policy,
            fit=CatalogFit(
                status=FIT_RUNNING,
                latest_epoch=None,
                max_epochs=self._max_epochs,
            ),
            current=CatalogCurrent(files=[], last=None),
            history={},
        )

    def write_catalog(self, catalog: CatalogDocument) -> None:
        parent = Directory.from_absolute(
            path=self._root, policy=DirectoryParsePolicy.CREATE
        )
        write_object(
            directory=parent,
            name=CATALOG_FILENAME,
            payload=catalog.model_dump(mode="json"),
            form=JsonTextForm.DOCUMENT,
        )

    def stamp_fit(self, catalog: CatalogDocument, *, epoch: int) -> CatalogDocument:
        return catalog.model_copy(
            update={
                "fit": CatalogFit(
                    status=FIT_RUNNING,
                    latest_epoch=epoch,
                    max_epochs=self._max_epochs,
                ),
                "monitor": self._monitor,
                "policy": self._policy,
            }
        )

    def commit_new(
        self,
        catalog: CatalogDocument,
        *,
        epoch: int,
        score: float,
        checkpoint_id: str,
        rel_path: str,
        written: File,
    ) -> CatalogDocument:
        digest = hash_file(written.path)
        size = written.path.stat().st_size
        current_files = list(catalog.current.files)
        new_row = CatalogLiveRow(
            epoch=epoch,
            id=checkpoint_id,
            path=rel_path,
            score=score,
            sha256=digest,
            bytes=size,
            rank=1,
        )
        current_files.append(new_row)
        scored: list[ScoredCheckpoint] = []
        n_current = len(current_files)
        by_id: dict[str, CatalogLiveRow] = {}
        for i in range(n_current):
            row = current_files[i]
            by_id[row.id] = row
            scored.append(
                ScoredCheckpoint(id=row.id, score=row.score, epoch=row.epoch)
            )
        last_id = checkpoint_id if self._policy.save_last else None
        last_epoch = epoch if self._policy.save_last else None
        result = apply_policy(
            files=scored,
            policy=self._policy,
            mode=self._monitor.mode,
            last_id=last_id,
            last_epoch=last_epoch,
        )
        self._prune_live(set(result.keep_ids))
        history = dict(catalog.history)
        history_keys = list(history)
        n_hist = len(history_keys)
        for i in range(n_hist):
            key = history_keys[i]
            entry = history[key]
            history[key] = entry.model_copy(
                update={"retained": entry.id in result.keep_ids}
            )
        rank_then = _rank_of(result.ordered, checkpoint_id)
        history[str(epoch)] = CatalogHistoryEntry(
            id=checkpoint_id,
            score=score,
            rank_then=rank_then,
            sha256=digest,
            retained=checkpoint_id in result.keep_ids,
        )
        ranked_rows: list[CatalogLiveRow] = []
        n_ranked = len(result.ranked)
        for i in range(n_ranked):
            ranked = result.ranked[i]
            ranked_rows.append(
                by_id[ranked.id].model_copy(update={"rank": ranked.rank})
            )
        return catalog.model_copy(
            update={
                "current": CatalogCurrent(
                    files=ranked_rows, last=result.last
                ),
                "history": history,
                "fit": CatalogFit(
                    status=FIT_RUNNING,
                    latest_epoch=epoch,
                    max_epochs=self._max_epochs,
                ),
                "monitor": self._monitor,
                "policy": self._policy,
            }
        )

    def fetch(
        self, request: CheckpointRequest, *, dest: Directory
    ) -> CheckpointRecord:
        if dest.spec != DirectorySpec.EXISTS:
            raise ValueError(f"fetch dest Directory must exist: {dest.path}")
        catalog = self.read_catalog()
        row = self._resolve(catalog, request)
        source = File.from_absolute(path=self._root / row.path)
        copied_path = dest.path / source.path.name
        shutil.copy2(source.path, copied_path)
        copied = File.from_absolute(path=copied_path)
        digest = hash_file(copied.path)
        size = copied.path.stat().st_size
        if digest != row.sha256 or size != row.bytes:
            copied.path.unlink()
            raise CheckpointLookupError(
                f"checksum mismatch for epoch {row.epoch} at {source.path}"
            )
        return CheckpointRecord(
            epoch=row.epoch,
            path=copied.path,
            rank=row.rank,
            score=row.score,
            kind=request.kind,
            checkpoint_id=row.id,
        )

    def current_scored(
        self, catalog: CatalogDocument
    ) -> tuple[ScoredCheckpoint, ...]:
        files = catalog.current.files
        n = len(files)
        scored: list[ScoredCheckpoint] = []
        for i in range(n):
            row = files[i]
            scored.append(
                ScoredCheckpoint(id=row.id, score=row.score, epoch=row.epoch)
            )
        return tuple(scored)

    def best_epoch(self, catalog: CatalogDocument) -> int | None:
        files = catalog.current.files
        n = len(files)
        for i in range(n):
            if files[i].rank == 1:
                return files[i].epoch
        return None

    def _prune_live(self, keep_ids: set[str]) -> None:
        if not self._live.is_dir():
            return
        live_files = list(self._live.glob("*.ckpt"))
        n_live = len(live_files)
        for i in range(n_live):
            live_path = live_files[i]
            if live_path.stem not in keep_ids:
                live_path.unlink()

    def _resolve(
        self,
        catalog: CatalogDocument,
        request: CheckpointRequest,
    ) -> CatalogLiveRow:
        files = catalog.current.files
        if request.kind == KIND_LAST:
            last = catalog.current.last
            if last is None:
                raise CheckpointLookupError("selector 'last' is not in the catalog")
            return self._row_by_id(files, last, catalog)
        if request.kind == KIND_RANK:
            rank = request.rank
            if rank is None:
                raise CheckpointLookupError("rank request is missing rank")
            return _match_rank(files=files, rank=rank)
        if request.kind == KIND_EPOCH:
            epoch = request.epoch
            if epoch is None:
                raise CheckpointLookupError("epoch request is missing epoch")
            return _match_epoch(files=files, epoch=epoch)
        if request.kind == KIND_VALUE:
            value = request.value
            if value is None:
                raise CheckpointLookupError("value request is missing value")
            return _match_score(files=files, value=value)
        raise CheckpointLookupError(f"unknown request kind {request.kind!r}")

    def _row_by_id(
        self,
        files: list[CatalogLiveRow],
        last: CatalogLastPointer,
        catalog: CatalogDocument,
    ) -> CatalogLiveRow:
        n = len(files)
        for i in range(n):
            if files[i].id == last.id:
                return files[i]
        entry = catalog.history.get(str(last.epoch))
        if entry is None or entry.id != last.id:
            raise CheckpointLookupError("selector 'last' is not in the catalog")
        live_file = File.from_absolute(path=self._live / f"{last.id}.ckpt")
        return CatalogLiveRow(
            epoch=last.epoch,
            id=last.id,
            path=f"{LIVE_DIRNAME}/{last.id}.ckpt",
            score=entry.score,
            sha256=entry.sha256,
            bytes=live_file.path.stat().st_size,
            rank=None,
        )


def _rank_of(ordered: tuple[RankedCheckpoint, ...], checkpoint_id: str) -> int:
    n = len(ordered)
    for i in range(n):
        if ordered[i].id == checkpoint_id:
            return ordered[i].rank
    return n


def _match_rank(
    *, files: list[CatalogLiveRow], rank: int
) -> CatalogLiveRow:
    n = len(files)
    for i in range(n):
        if files[i].rank == rank:
            return files[i]
    raise CheckpointLookupError(f"selector 'rank={rank}' is not in the catalog")


def _match_epoch(
    *, files: list[CatalogLiveRow], epoch: int
) -> CatalogLiveRow:
    n = len(files)
    for i in range(n):
        if files[i].epoch == epoch:
            return files[i]
    raise CheckpointLookupError(
        f"selector 'epoch={epoch}' is not in the catalog"
    )


def _match_score(
    *, files: list[CatalogLiveRow], value: float
) -> CatalogLiveRow:
    n = len(files)
    for i in range(n):
        if files[i].score == value:
            return files[i]
    raise CheckpointLookupError(
        f"selector 'value={value}' is not in the catalog"
    )
