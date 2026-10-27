"""Checkpoint policy, Lightning callback, catalog, inspect lookup."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from dl_toolbox.base.input_output.json_io import read_json, write_json
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.dl_engine.training.config import MonitorConfig

_EPOCH_NAME = re.compile(r"^epoch=(\d+)\.ckpt$")
_CATALOG_FILENAME = "catalog.json"
CHECKPOINTS_DIRNAME = "checkpoints"


@dataclass(frozen=True)
class CheckpointLookup:
    epoch: int
    path: Path
    rank: int | None
    kind: str


class CheckpointLookupError(ValueError):
    """Selector does not resolve to a file on disk."""


class CheckpointManager:
    def __init__(
        self,
        *,
        save_top_k: int,
        save_last: bool,
        run_dir: Path,
        monitor: MonitorConfig,
    ) -> None:
        if save_top_k < -1:
            raise ValueError(f"save_top_k must be >= -1, got {save_top_k}")
        if monitor.mode not in ("min", "max"):
            raise ValueError(f"monitor.mode must be 'min' or 'max', got {monitor.mode!r}")
        if not run_dir.is_absolute():
            raise ValueError(f"run_dir must be an absolute path, got {run_dir}")
        self.save_top_k = save_top_k
        self.save_last = bool(save_last)
        self.run_dir = run_dir
        self.monitor = monitor
        self.dirpath = run_dir / CHECKPOINTS_DIRNAME
        self.catalog_path = self.dirpath / _CATALOG_FILENAME

    def as_callback(self) -> ModelCheckpoint:
        self.dirpath.mkdir(parents=True, exist_ok=True)
        return _CatalogCheckpoint(self)

    def lookup(self, selector: str) -> CheckpointLookup:
        text = parse_non_empty_str(selector, label="selector")
        catalog = self._read_catalog()
        files = list(catalog.get("files", []))
        last = catalog.get("last")
        if text == "last":
            if last is None:
                raise CheckpointLookupError("selector 'last' is not in the catalog")
            return self._existing(last, kind="last", rank=None)
        if text.startswith("rank="):
            rank = _parse_selector_int(text, label="rank")
            if rank < 1:
                raise CheckpointLookupError(f"rank must be >= 1, got {rank}")
            ranked = [row for row in files if row.get("rank") == rank]
            if not ranked:
                raise CheckpointLookupError(f"selector {text!r} is not in the catalog")
            return self._existing(ranked[0], kind="rank", rank=rank)
        if text.startswith("epoch="):
            epoch = _parse_selector_int(text, label="epoch")
            matched = [row for row in files if int(row["epoch"]) == epoch]
            if not matched:
                raise CheckpointLookupError(f"selector {text!r} is not in the catalog")
            return self._existing(matched[0], kind="epoch", rank=matched[0].get("rank"))
        raise CheckpointLookupError(
            f"selector must be 'last', 'rank=N', or 'epoch=N', got {text!r}"
        )

    def _existing(self, row: dict[str, Any], *, kind: str, rank: int | None) -> CheckpointLookup:
        path = self.dirpath / str(row["path"])
        if not path.is_file():
            raise CheckpointLookupError(f"catalog entry is missing on disk: {path}")
        return CheckpointLookup(
            epoch=int(row["epoch"]),
            path=path,
            rank=rank if rank is not None else row.get("rank"),
            kind=kind,
        )

    def _read_catalog(self) -> dict[str, Any]:
        if not self.catalog_path.is_file():
            raise CheckpointLookupError(f"catalog does not exist: {self.catalog_path}")
        return read_json(self.catalog_path)

    def write_catalog(self, trainer: L.Trainer, callback: ModelCheckpoint) -> None:
        files: list[dict[str, Any]] = []
        scored = []
        best = getattr(callback, "best_k_models", None) or {}
        for path_str, score in best.items():
            path = Path(path_str)
            epoch = _epoch_from_name(path.name)
            if epoch is None:
                continue
            scored.append((float(score), epoch, path.name))
        reverse = self.monitor.mode == "max"
        scored.sort(key=lambda item: item[0], reverse=reverse)
        n = len(scored)
        for i in range(n):
            score, epoch, filename = scored[i]
            files.append(
                {
                    "epoch": epoch,
                    "path": filename,
                    "score": score,
                    "rank": i + 1,
                }
            )
        last_row = None
        last_path = getattr(callback, "last_model_path", "") or ""
        if last_path:
            last_file = Path(last_path)
            epoch = _epoch_from_name(last_file.name)
            if epoch is None:
                epoch = int(trainer.current_epoch)
            last_row = {"epoch": epoch, "path": last_file.name}
        payload = {
            "monitor": {"key": self.monitor.key, "mode": self.monitor.mode},
            "save_top_k": self.save_top_k,
            "save_last": self.save_last,
            "files": files,
            "last": last_row,
        }
        write_json(self.catalog_path, payload)


def _parse_selector_int(text: str, *, label: str) -> int:
    suffix = text.split("=", 1)[1]
    try:
        return int(suffix)
    except ValueError as exc:
        raise CheckpointLookupError(f"{label} selector must be an int, got {text!r}") from exc


def _epoch_from_name(name: str) -> int | None:
    matched = _EPOCH_NAME.match(name)
    if matched is None:
        return None
    return int(matched.group(1))


class _CatalogCheckpoint(ModelCheckpoint):
    def __init__(self, manager: CheckpointManager) -> None:
        super().__init__(
            dirpath=str(manager.dirpath),
            filename="epoch={epoch:03d}",
            auto_insert_metric_name=False,
            monitor=manager.monitor.key,
            mode=manager.monitor.mode,
            save_top_k=manager.save_top_k,
            save_last=manager.save_last,
        )
        self._toolbox_manager = manager

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        super().on_validation_end(trainer, pl_module)
        if trainer.sanity_checking:
            return
        self._toolbox_manager.write_catalog(trainer, self)
